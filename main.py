from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional, List
import os
import bcrypt
import asyncio
import httpx
import json
import random
import simulation_engine as sim

# CONFIG
SECRET_KEY = "SUPER_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

DATABASE_URL = os.getenv("DATABASE_URL")

# IA / OpenAI (clave sólo por variable de entorno, nunca en código)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Pesos del modelo de prioridad multi-factor
PRIORITY_WEIGHTS = {
    "impacto_ciudadano": 0.35,
    "urgencia_temporal": 0.25,
    "riesgo_seguridad": 0.20,
    "vulnerabilidad_poblacion": 0.10,
    "reincidencia_probable": 0.10,
}

DEFAULT_PRIORITY_FACTORS = {
    "impacto_ciudadano": 50,
    "urgencia_temporal": 50,
    "riesgo_seguridad": 50,
    "vulnerabilidad_poblacion": 50,
    "reincidencia_probable": 50,
}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"sslmode": "require"}  # IMPORTANTE para Render Postgres
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

app = FastAPI()

# ─── Start simulation engine on startup ──────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    sim.start_simulation(asyncio.get_event_loop())

# ─── Fleet WebSocket ──────────────────────────────────────────────────────────

@app.websocket("/ws/fleet")
async def fleet_ws(websocket: WebSocket):
    await websocket.accept()
    sim.register_ws(websocket)
    # Send current state immediately on connect
    await websocket.send_text(__import__("json").dumps(sim.get_current_state()))
    try:
        while True:
            # Keep the connection alive; actual pushes come from the simulation loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sim.unregister_ws(websocket)

# ─── Fleet HTTP polling fallback ──────────────────────────────────────────────

@app.get("/api/fleet/state")
def fleet_state():
    return sim.get_current_state()

# CORS - Permitir frontend en cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: ["https://vita360.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# MODELOS DB

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    role = Column(String)

class Area(Base):
    __tablename__ = "areas"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    sla_hours = Column(Integer)

class Squad(Base):
    __tablename__ = "squads"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    area_name = Column(String, nullable=True)
    pending_tasks = Column(Integer, default=0)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(Text)
    priority_score = Column(Integer)
    urgency_level = Column(String)
    status = Column(String)
    planned_date = Column(DateTime)
    area_id = Column(Integer, ForeignKey("areas.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    assigned_to = Column(Integer, nullable=True)
    squad_name = Column(String, nullable=True)           # ← nuevo: nombre cuadrilla asignada
    created_at = Column(DateTime, default=datetime.utcnow)
    # Geolocalización
    lat = Column(Float, nullable=True)                   # ← nuevo
    lng = Column(Float, nullable=True)                   # ← nuevo
    # Multi-factor metrics y pesos de prioridad (almacenados como JSON serializado)
    metrics_json = Column(Text, nullable=True)
    priority_weights = Column(Text, nullable=True)

class Evidence(Base):
    __tablename__ = "evidence"
    id = Column(Integer, primary_key=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"))
    image_url = Column(String)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# UTILIDADES

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str):
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub")
        user_id = int(user_id_str)
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token: user not found")
        return user
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token: invalid user ID format")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation error: {str(e)}")

# MOTOR DE CLASIFICACIÓN

def classify_ticket(description):
    description = description.lower()
    if "árbol" in description:
        return "Áreas Verdes", 90
    if "basura" in description or "contenedor" in description:
        return "Aseo", 70
    if "vereda" in description or "hoyo" in description:
        return "Infraestructura", 80
    return "Atención General", 50

def calculate_urgency(score):
    if score >= 85:
        return "Alta"
    if score >= 60:
        return "Media"
    return "Baja"


# ─── IA (OpenAI) centralizada en backend ──────────────────────────────────────

def _openai_available() -> bool:
    return bool(OPENAI_API_KEY)


def _openai_chat(messages, max_tokens: int = 60) -> str:
    if not _openai_available():
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada en el backend")

    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "max_tokens": max_tokens,
                "messages": messages,
            },
            timeout=20.0,
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error conectando a OpenAI: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error OpenAI: {response.status_code} {response.text}")

    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not content:
        raise HTTPException(status_code=502, detail="Respuesta vacía de OpenAI")
    return content


def classify_ticket_with_ai(title: str, description: str) -> str:
    if not _openai_available():
        area, _ = classify_ticket(description)
        return area

    content = _openai_chat(
        [
            {
                "role": "system",
                "content": (
                    "Eres un clasificador de solicitudes municipales. "
                    "Según el título y la descripción, responde SOLO el nombre del área más adecuada "
                    "entre opciones típicas como: \"Áreas Verdes\", \"Aseo\", \"Infraestructura\", "
                    "\"Atención General\" u otra similar, sin explicación adicional."
                ),
            },
            {
                "role": "user",
                "content": f"Título: {title}\nDescripción: {description}\nDevuelve solo el nombre del área.",
            },
        ],
        max_tokens=40,
    )

    area_name = content.splitlines()[0].strip()
    return area_name


def calculate_priority_factors_with_ai(title: str, description: str) -> dict:
    if not _openai_available():
        return DEFAULT_PRIORITY_FACTORS.copy()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a municipal priority evaluation engine.\n"
                "Return ONLY valid JSON with numeric fields (0-100 integers) and no additional text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate this municipal report and return:\n\n"
                "{\n"
                '  "impacto_ciudadano": number,\n'
                '  "urgencia_temporal": number,\n'
                '  "riesgo_seguridad": number,\n'
                '  "vulnerabilidad_poblacion": number,\n'
                '  "reincidencia_probable": number\n'
                "}\n\n"
                f"Title: {title}\n"
                f"Description: {description}"
            ),
        },
    ]

    try:
        raw = _openai_chat(messages, max_tokens=200)
    except HTTPException:
        return DEFAULT_PRIORITY_FACTORS.copy()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Respuesta de OpenAI no es JSON válido para factores de prioridad")

    expected_keys = [
        "impacto_ciudadano",
        "urgencia_temporal",
        "riesgo_seguridad",
        "vulnerabilidad_poblacion",
        "reincidencia_probable",
    ]

    factors: dict = {}
    for key in expected_keys:
        if key not in data:
            raise HTTPException(status_code=502, detail=f"Falta el campo '{key}' en la respuesta de OpenAI")
        value = data[key]
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=502, detail=f"El campo '{key}' no es un entero válido: {value!r}")
        if not (0 <= ivalue <= 100):
            raise HTTPException(status_code=502, detail=f"El campo '{key}' está fuera de rango 0–100: {ivalue}")
        factors[key] = ivalue

    return factors


# ─── Vitacura polygon helpers ─────────────────────────────────────────────────

# Polígono de la comuna de Vitacura (lon, lat)
VITACURA_POLYGON = [
    (-70.6061611, -33.4102650),
    (-70.6041870, -33.4034583),
    (-70.6041870, -33.3957911),
    (-70.5981789, -33.3894849),
    (-70.5933723, -33.3851849),
    (-70.5849609, -33.3812431),
    (-70.5748329, -33.3794513),
    (-70.5653229, -33.3770144),
    (-70.5573406, -33.3758676),
    (-70.5485001, -33.3742907),
    (-70.5423203, -33.3756500),
    (-70.5380249, -33.3807000),
    (-70.5360000, -33.3900000),
    (-70.5390000, -33.4050000),
    (-70.5500000, -33.4150000),
    (-70.5650000, -33.4200000),
    (-70.5850000, -33.4200000),
    (-70.6000000, -33.4160000),
    (-70.6061611, -33.4102650),
]

def _point_in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray-casting algorithm: returns True if point (x,y) is inside polygon."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _random_point_in_vitacura() -> tuple:
    """Return a random (lat, lng) strictly inside the Vitacura polygon."""
    lons = [p[0] for p in VITACURA_POLYGON]
    lats = [p[1] for p in VITACURA_POLYGON]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    for _ in range(1000):
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)
        if _point_in_polygon(lon, lat, VITACURA_POLYGON):
            return lat, lon
    # Fallback: centroid of Vitacura if somehow never lands inside
    return -33.3947, -70.5680


def compute_priority_score_from_factors(factors: dict, weights: dict) -> int:
    total = 0.0
    for key, weight in weights.items():
        total += float(factors.get(key, 0)) * float(weight)
    score = round(total)
    return max(0, min(100, score))

# SCHEMAS

class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str

class TicketCreate(BaseModel):
    title: str
    description: str
    # Foto integrada a la solicitud
    image_url: Optional[str] = None
    image_description: Optional[str] = ""
    # Geolocalización enviada por el frontend
    lat: Optional[float] = None
    lng: Optional[float] = None
    timestamp: Optional[str] = None

class TicketUpdate(BaseModel):
    status: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None

class AITicketPayload(BaseModel):
    title: str
    description: str

class AssignSquadRequest(BaseModel):
    squad_name: str

class UpdateStatusRequest(BaseModel):
    status: str

class AddEvidenceRequest(BaseModel):
    image_url: str
    description: str = ""

# ─── Helper: serializar ticket ────────────────────────────────────────────────

def _serialize_ticket(ticket: Ticket, db: Session, include_reporter: bool = False) -> dict:
    area = db.query(Area).filter(Area.id == ticket.area_id).first()
    assigned_user = db.query(User).filter(User.id == ticket.assigned_to).first() if ticket.assigned_to else None
    evidences = db.query(Evidence).filter(Evidence.ticket_id == ticket.id).all()

    result = {
        "id": ticket.id,
        "title": ticket.title,
        "description": ticket.description,
        "status": ticket.status,
        "urgency_level": ticket.urgency_level,
        "priority_score": ticket.priority_score,
        "area_name": area.name if area else "Sin asignar",
        "squad_name": ticket.squad_name,
        "assigned_to": assigned_user.name if assigned_user else None,
        "created_at": ticket.created_at,
        "planned_date": ticket.planned_date,
        "lat": ticket.lat,
        "lon": ticket.lng,   # el frontend espera 'lon' no 'lng'
        "evidences": [
            {
                "image_url": ev.image_url,
                "description": getattr(ev, "description", ""),
                "created_at": ev.created_at,
            }
            for ev in evidences
        ],
    }

    if include_reporter:
        reporter = db.query(User).filter(User.id == ticket.user_id).first()
        result["reported_by"] = reporter.name if reporter else None
        result["reported_by_email"] = reporter.email if reporter else None

    return result

# ENDPOINTS

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(user.password)
    new_user = User(name=user.name, email=user.email, password=hashed, role=user.role)
    db.add(new_user)
    db.commit()
    return {"message": "User created"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=400, detail="Incorrect credentials")

    token = create_access_token({"sub": str(user.id)})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "name": user.name,
        "id": user.id,
    }

# ─── TICKETS ──────────────────────────────────────────────────────────────────

@app.post("/tickets")
def create_ticket(
    ticket: TicketCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    area_name = classify_ticket_with_ai(ticket.title, ticket.description)
    area = db.query(Area).filter(Area.name == area_name).first()
    if not area:
        area = Area(name=area_name, sla_hours=72)
        db.add(area)
        db.commit()
        db.refresh(area)

    factors = calculate_priority_factors_with_ai(ticket.title, ticket.description)
    priority_score = compute_priority_score_from_factors(factors, PRIORITY_WEIGHTS)
    urgency = calculate_urgency(priority_score)
    planned_date = datetime.utcnow() + timedelta(hours=area.sla_hours)

    # Usar coordenadas del ciudadano o generar punto aleatorio dentro de Vitacura
    if ticket.lat is not None and ticket.lng is not None:
        ticket_lat, ticket_lng = ticket.lat, ticket.lng
    else:
        ticket_lat, ticket_lng = _random_point_in_vitacura()

    new_ticket = Ticket(
        title=ticket.title,
        description=ticket.description,
        priority_score=priority_score,
        urgency_level=urgency,
        status="Recibido",
        planned_date=planned_date,
        area_id=area.id,
        user_id=current_user.id,
        lat=ticket_lat,
        lng=ticket_lng,
        metrics_json=json.dumps(factors),
        priority_weights=json.dumps(PRIORITY_WEIGHTS),
    )

    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)

    # ─── Evidencia ────────────────────────────────────────────────────────────
    evidence_id = None
    if ticket.image_url:
        ev = Evidence(
            ticket_id=new_ticket.id,
            image_url=ticket.image_url,
            description=(ticket.image_description or ""),
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
        evidence_id = ev.id

    return {
        "id": new_ticket.id,
        "ticket_id": new_ticket.id,
        "area": area.name,
        "priority": priority_score,
        "urgency_level": urgency,
        "planned_date": planned_date,
        "evidence_id": evidence_id,
        "metrics": {
            "riesgo_seguridad": factors.get("riesgo_seguridad", 50),
            "urgencia": factors.get("urgencia_temporal", 50),
            "impacto": factors.get("impacto_ciudadano", 50),
            "sla_legal": 50,
            "vulnerabilidad_lugar": factors.get("vulnerabilidad_poblacion", 50),
        },
        "location_context": {
            "near_school": False,
            "near_hospital": False,
            "near_high_traffic": False,
            "in_critical_zone": False,
        },
        "reasoning": f"Ticket clasificado en área '{area.name}' con prioridad {urgency} (score {priority_score}/100).",
    }

@app.get("/my-tickets")
def my_tickets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tickets = db.query(Ticket).filter(Ticket.user_id == current_user.id).all()
    return [_serialize_ticket(t, db) for t in tickets]

@app.get("/tickets/count")
def get_tickets_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Endpoint ligero para el monitor de IA.
    Devuelve solo el total de tickets sin serializar nada.
    Solo accesible por operadores/supervisores."""
    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden acceder")
    count = db.query(Ticket).count()
    return {"count": count}

@app.get("/tickets")
def get_tickets(
    status: Optional[str] = None,
    area: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    order: Optional[str] = "desc",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden acceder")

    query = db.query(Ticket)
    if order == "asc":
        query = query.order_by(Ticket.id.asc())
    else:
        query = query.order_by(Ticket.priority_score.desc(), Ticket.id.desc())

    if status:
        query = query.filter(Ticket.status == status)
    if area:
        area_obj = db.query(Area).filter(Area.name == area).first()
        if area_obj:
            query = query.filter(Ticket.area_id == area_obj.id)

    if offset:
        query = query.offset(offset)
    if limit:
        query = query.limit(limit)

    tickets = query.all()
    return [_serialize_ticket(t, db, include_reporter=True) for t in tickets]

@app.get("/tickets/{ticket_id}")
def get_ticket(
    ticket_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Ciudadanos solo ven sus propios tickets
    if current_user.role not in ["operador", "operator", "supervisor"] and ticket.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este ticket")

    return _serialize_ticket(ticket, db, include_reporter=True)

@app.patch("/tickets/{ticket_id}")
def update_ticket(
    ticket_id: int,
    data: TicketUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if data.status is not None:
        ticket.status = data.status
    if data.title is not None:
        ticket.title = data.title
    if data.description is not None:
        ticket.description = data.description

    db.commit()
    db.refresh(ticket)
    return _serialize_ticket(ticket, db, include_reporter=True)

@app.patch("/tickets/{ticket_id}/status")
def update_status(
    ticket_id: int,
    request: UpdateStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.status = request.status
    db.commit()
    return {"message": "Status updated", "new_status": request.status}

@app.delete("/tickets/{ticket_id}")
def delete_ticket(
    ticket_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Solo admins/operadores o el dueño pueden eliminar
    if current_user.role not in ["operador", "operator", "supervisor"] and ticket.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tienes permiso para eliminar este ticket")

    db.delete(ticket)
    db.commit()
    return {"message": "Ticket deleted"}

# ─── ASIGNACIÓN DE CUADRILLA ──────────────────────────────────────────────────

@app.post("/tickets/{ticket_id}/assign")
def assign_squad(
    ticket_id: int,
    request: AssignSquadRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden asignar cuadrillas")

    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.squad_name = request.squad_name
    if ticket.status == "Recibido":
        ticket.status = "Asignado"

    db.commit()
    db.refresh(ticket)
    return {
        "message": "Cuadrilla asignada",
        "squad_name": ticket.squad_name,
        "status": ticket.status,
    }

# ─── EVIDENCIA ────────────────────────────────────────────────────────────────

@app.post("/tickets/{ticket_id}/evidence")
def add_evidence(
    ticket_id: int,
    request: AddEvidenceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    existing_count = db.query(Evidence).filter(Evidence.ticket_id == ticket_id).count()
    if existing_count >= 1:
        raise HTTPException(status_code=400, detail="Este ticket ya tiene una foto asociada")

    evidence = Evidence(
        ticket_id=ticket_id,
        image_url=request.image_url,
        description=request.description,
    )
    db.add(evidence)
    db.commit()
    db.refresh(evidence)
    return {
        "message": "Evidence added",
        "evidence_id": evidence.id,
        "id": evidence.id,
        "ticket_id": evidence.ticket_id,
        "image_url": evidence.image_url,
        "description": evidence.description,
        "created_at": evidence.created_at,
    }

@app.get("/tickets/{ticket_id}/evidence")
def get_evidence(
    ticket_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    evidences = db.query(Evidence).filter(Evidence.ticket_id == ticket_id).all()
    return [
        {
            "id": ev.id,
            "ticket_id": ev.ticket_id,
            "image_url": ev.image_url,
            "description": ev.description,
            "created_at": ev.created_at,
        }
        for ev in evidences
    ]

@app.delete("/evidence/{evidence_id}")
def delete_evidence(
    evidence_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ev = db.query(Evidence).filter(Evidence.id == evidence_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Evidence not found")
    db.delete(ev)
    db.commit()
    return {"message": "Evidence deleted"}

# ─── CUADRILLAS ───────────────────────────────────────────────────────────────

@app.get("/squads")
def get_squads(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    squads = db.query(Squad).all()

    # Si no hay cuadrillas registradas, devolver set por defecto
    if not squads:
        default_squads = [
            {"id": 1, "name": "Cuadrilla Áreas Verdes A", "area_name": "Áreas Verdes", "pending_tasks": 0},
            {"id": 2, "name": "Cuadrilla Áreas Verdes B", "area_name": "Áreas Verdes", "pending_tasks": 0},
            {"id": 3, "name": "Cuadrilla Infraestructura", "area_name": "Infraestructura", "pending_tasks": 0},
            {"id": 4, "name": "Cuadrilla Aseo", "area_name": "Aseo", "pending_tasks": 0},
            {"id": 5, "name": "Cuadrilla General", "area_name": "Atención General", "pending_tasks": 0},
        ]
        return default_squads

    return [
        {
            "id": s.id,
            "name": s.name,
            "area_name": s.area_name,
            "pending_tasks": s.pending_tasks,
        }
        for s in squads
    ]

# ─── ÁREAS ────────────────────────────────────────────────────────────────────

@app.get("/areas")
def get_areas(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    areas = db.query(Area).all()
    return [
        {
            "id": a.id,
            "name": a.name,
            "sla_hours": a.sla_hours,
        }
        for a in areas
    ]

# ─── ESTADÍSTICAS ─────────────────────────────────────────────────────────────

@app.get("/stats/dashboard")
def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    all_tickets = db.query(Ticket).all()
    total = len(all_tickets)

    open_tickets = [t for t in all_tickets if t.status not in ("Resuelto", "Cerrado")]
    resolved = [t for t in all_tickets if t.status in ("Resuelto", "Cerrado")]

    at_risk = sum(
        1 for t in open_tickets if t.urgency_level == "Alta"
    )

    # Tiempo promedio de resolución (horas)
    response_times = []
    for t in resolved:
        if t.created_at and t.planned_date:
            delta = (t.planned_date - t.created_at).total_seconds() / 3600
            response_times.append(delta)
    avg_hours = sum(response_times) / len(response_times) if response_times else 0
    avg_response = f"{int(avg_hours)}h {int((avg_hours % 1) * 60)}m"

    resolved_first_contact_pct = round((len(resolved) / total * 100) if total > 0 else 0)

    return {
        "total_open": len(open_tickets),
        "total_tickets": total,
        "resolved_first_contact_percentage": resolved_first_contact_pct,
        "avg_response_time": avg_response,
        "at_risk_count": at_risk,
        "tickets_at_risk": at_risk,
        "tickets_by_status": {
            status: sum(1 for t in all_tickets if t.status == status)
            for status in ["Recibido", "Asignado", "En Gestión", "Resuelto", "Cerrado"]
        },
    }

@app.get("/stats/areas")
def get_area_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    all_tickets = db.query(Ticket).all()
    result: dict = {}

    for ticket in all_tickets:
        area = db.query(Area).filter(Area.id == ticket.area_id).first()
        area_name = area.name if area else "Sin asignar"
        if area_name not in result:
            result[area_name] = {"total": 0, "open": 0, "resolved": 0, "at_risk": 0}
        result[area_name]["total"] += 1
        if ticket.status in ("Resuelto", "Cerrado"):
            result[area_name]["resolved"] += 1
        else:
            result[area_name]["open"] += 1
            if ticket.urgency_level == "Alta":
                result[area_name]["at_risk"] += 1

    return result

# ─── ENDPOINTS IA para frontend (monitor operador) ────────────────────────────

@app.post("/ai/tickets/classify")
def ai_classify_ticket(
    payload: AITicketPayload,
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden acceder a IA de clasificación")

    area = classify_ticket_with_ai(payload.title, payload.description)
    factors = calculate_priority_factors_with_ai(payload.title, payload.description)
    score = compute_priority_score_from_factors(factors, PRIORITY_WEIGHTS)
    urgency = calculate_urgency(score)
    return {
        "area": area,
        "score": score,
        "urgency": urgency,
        "metrics": factors,
        "weights": PRIORITY_WEIGHTS,
    }


@app.post("/ai/tickets/priority")
def ai_ticket_priority(
    payload: AITicketPayload,
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden acceder a IA de prioridad")

    factors = calculate_priority_factors_with_ai(payload.title, payload.description)
    score = compute_priority_score_from_factors(factors, PRIORITY_WEIGHTS)
    urgency = calculate_urgency(score)
    return {
        "score": score,
        "urgency": urgency,
        "metrics": factors,
        "weights": PRIORITY_WEIGHTS,
    }


# ─── Nuevo endpoint: descripción de tarea + tiempo estimado ──────────────────

class AITaskPayload(BaseModel):
    title: str
    description: str
    area: str
    squad_types: List[str]   # nombres de cuadrillas disponibles en el área


@app.post("/ai/tickets/task")
def ai_ticket_task(
    payload: AITaskPayload,
    current_user: User = Depends(get_current_user),
):
    """Genera una descripción concisa de la tarea y el tiempo estimado de resolución.
    Recibe el área clasificada y los tipos de cuadrillas disponibles como contexto."""

    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden acceder")

    squad_list = ", ".join(payload.squad_types) if payload.squad_types else "cuadrilla general"

    if not _openai_available():
        # Fallback determinista: resumen = primeras 10 palabras del título
        words = payload.title.split()
        summary = " ".join(words[:10]) + ("…" if len(words) > 10 else "")
        return {"task_summary": summary, "estimated_hours": 24}

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente municipal. Dado un reporte ciudadano y el área de gestión, "
                "responde SOLO con JSON válido con dos campos:\n"
                '{"task_summary": "<descripción de la tarea en máximo 15 palabras>", "estimated_hours": <número entero de horas>}\n'
                "No incluyas texto adicional fuera del JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Área: {payload.area}\n"
                f"Cuadrillas disponibles: {squad_list}\n"
                f"Título del reporte: {payload.title}\n"
                f"Descripción: {payload.description}\n\n"
                "Devuelve JSON con task_summary (acción concreta para la cuadrilla) y estimated_hours."
            ),
        },
    ]

    try:
        raw = _openai_chat(messages, max_tokens=100)
    except HTTPException:
        words = payload.title.split()
        summary = " ".join(words[:10]) + ("…" if len(words) > 10 else "")
        return {"task_summary": summary, "estimated_hours": 24}

    try:
        data = json.loads(raw)
        summary = str(data.get("task_summary", payload.title[:60]))
        hours = int(data.get("estimated_hours", 24))
        return {"task_summary": summary, "estimated_hours": max(1, min(hours, 720))}
    except (json.JSONDecodeError, ValueError):
        words = payload.title.split()
        summary = " ".join(words[:10]) + ("…" if len(words) > 10 else "")
        return {"task_summary": summary, "estimated_hours": 24}

