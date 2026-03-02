from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Float, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
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

# ─── Configuración de áreas: SLA (horas) y número de cuadrillas ───────────────

AREA_CONFIG: dict[str, dict] = {
    "Atención General":        {"sla_hours": 72, "squad_count": 2},
    "Infraestructura Vial":    {"sla_hours": 48, "squad_count": 3},
    "Aseo y Ornato":           {"sla_hours": 24, "squad_count": 3},
    "Alumbrado Público":       {"sla_hours": 24, "squad_count": 2},
    "Áreas Verdes y Arbolado": {"sla_hours": 48, "squad_count": 3},
    "Seguridad y Emergencias": {"sla_hours": 4,  "squad_count": 2},
    "Agua y Alcantarillado":   {"sla_hours": 12, "squad_count": 2},
    "Tránsito y Señalización": {"sla_hours": 48, "squad_count": 2},
    "Residuos y Reciclaje":    {"sla_hours": 24, "squad_count": 2},
    "Mantenimiento Urbano":    {"sla_hours": 48, "squad_count": 3},
    "Aseo":                    {"sla_hours": 72, "squad_count": 2},
}

# ─── Límites geográficos de la comuna de Vitacura ────────────────────────────

VITACURA_BOUNDS = {
    "lat_min": -33.4180,
    "lat_max": -33.3560,
    "lon_min": -70.6250,
    "lon_max": -70.5490,
}

def random_vitacura_coords() -> tuple[float, float]:
    lat = random.uniform(VITACURA_BOUNDS["lat_min"], VITACURA_BOUNDS["lat_max"])
    lon = random.uniform(VITACURA_BOUNDS["lon_min"], VITACURA_BOUNDS["lon_max"])
    return round(lat, 6), round(lon, 6)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"sslmode": "require"}  # IMPORTANTE para Render Postgres
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

app = FastAPI()

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
    """Cuadrilla de trabajo asociada a un área."""
    __tablename__ = "squads"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    area_name = Column(String)   # denormalizado para búsqueda rápida
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
    created_at = Column(DateTime, default=datetime.utcnow)
    # Multi-factor metrics y pesos de prioridad (almacenados como JSON serializado)
    metrics_json = Column(Text, nullable=True)
    priority_weights = Column(Text, nullable=True)
    # Geolocalización dentro de Vitacura
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    # Cuadrilla auto-asignada
    squad_name = Column(String, nullable=True)

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

# 🔥 HASH SIN PASSLIB

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

def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="login")), db: Session = Depends(get_db)):
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

# ─── Squad helpers ────────────────────────────────────────────────────────────

def get_or_create_area(area_name: str, db: Session) -> "Area":
    """Devuelve el área existente o la crea con SLA de AREA_CONFIG."""
    area = db.query(Area).filter(Area.name == area_name).first()
    if not area:
        cfg = AREA_CONFIG.get(area_name, {"sla_hours": 72})
        area = Area(name=area_name, sla_hours=cfg["sla_hours"])
        db.add(area)
        db.commit()
        db.refresh(area)
    else:
        # Sincronizar SLA si el área ya existe y está en AREA_CONFIG
        cfg = AREA_CONFIG.get(area_name)
        if cfg and area.sla_hours != cfg["sla_hours"]:
            area.sla_hours = cfg["sla_hours"]
            db.commit()
    return area


def assign_squad_to_ticket(area_name: str, db: Session) -> str | None:
    """
    Asigna la cuadrilla del área con menos tareas pendientes.
    En caso de empate, elige aleatoriamente entre las empatadas.
    Incrementa el contador de la cuadrilla elegida.
    """
    squads = db.query(Squad).filter(Squad.area_name == area_name).all()
    if not squads:
        return None
    min_tasks = min(s.pending_tasks for s in squads)
    candidates = [s for s in squads if s.pending_tasks == min_tasks]
    chosen = random.choice(candidates)
    chosen.pending_tasks += 1
    db.commit()
    return chosen.name


def release_squad_task(squad_name: str, db: Session):
    """Decrementa el contador de tareas pendientes de la cuadrilla."""
    if not squad_name:
        return
    squad = db.query(Squad).filter(Squad.name == squad_name).first()
    if squad and squad.pending_tasks > 0:
        squad.pending_tasks -= 1
        db.commit()


def seed_squads(db: Session):
    """Crea las cuadrillas definidas en AREA_CONFIG si no existen."""
    for area_name, cfg in AREA_CONFIG.items():
        for i in range(1, cfg["squad_count"] + 1):
            squad_name = f"{area_name} – Cuadrilla {i}"
            exists = db.query(Squad).filter(Squad.name == squad_name).first()
            if not exists:
                db.add(Squad(name=squad_name, area_name=area_name, pending_tasks=0))
    db.commit()


def _run_migrations():
    """Agrega columnas nuevas a tablas existentes (idempotente)."""
    migrations = [
        "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS lat FLOAT",
        "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS lon FLOAT",
        "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS squad_name VARCHAR",
    ]
    with engine.connect() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass


# ─── Start simulation engine on startup ──────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    _run_migrations()
    db = SessionLocal()
    try:
        seed_squads(db)
    finally:
        db.close()
    sim.start_simulation(asyncio.get_event_loop())

# ─── Fleet WebSocket ──────────────────────────────────────────────────────────

@app.websocket("/ws/fleet")
async def fleet_ws(websocket: WebSocket):
    await websocket.accept()
    sim.register_ws(websocket)
    await websocket.send_text(__import__("json").dumps(sim.get_current_state()))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sim.unregister_ws(websocket)

# ─── Fleet HTTP polling fallback ──────────────────────────────────────────────

@app.get("/api/fleet/state")
def fleet_state():
    return sim.get_current_state()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# MOTOR DE CLASIFICACIÓN

def classify_ticket(description):
    description = description.lower()
    if "árbol" in description:
        return "Áreas Verdes y Arbolado", 90
    if "basura" in description or "contenedor" in description:
        return "Aseo y Ornato", 70
    if "vereda" in description or "hoyo" in description:
        return "Infraestructura Vial", 80
    if "alumbrado" in description or "luminaria" in description or "luz" in description:
        return "Alumbrado Público", 75
    if "agua" in description or "alcantarillado" in description:
        return "Agua y Alcantarillado", 80
    if "tránsito" in description or "señal" in description or "semáforo" in description:
        return "Tránsito y Señalización", 65
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

    area_names = list(AREA_CONFIG.keys())
    content = _openai_chat(
        [
            {
                "role": "system",
                "content": (
                    "Eres un clasificador de solicitudes municipales de Vitacura, Chile. "
                    "Según el título y la descripción, responde SOLO el nombre exacto del área "
                    f"entre estas opciones: {', '.join(area_names)}. Sin explicación adicional."
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
    # Validar que sea un área conocida, si no, usar Atención General
    if area_name not in AREA_CONFIG:
        area_name = "Atención General"
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

    factors: dict[str, int] = {}
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
    image_url: str | None = None
    image_description: str | None = ""


class AITicketPayload(BaseModel):
    title: str
    description: str

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
        "id": user.id
    }

@app.post("/tickets")
def create_ticket(
    ticket: TicketCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Clasificación de área usando IA si está disponible (fallback heurístico)
    area_name = classify_ticket_with_ai(ticket.title, ticket.description)
    area = get_or_create_area(area_name, db)

    # Factores de prioridad multi-factor (IA o valores por defecto)
    factors = calculate_priority_factors_with_ai(ticket.title, ticket.description)
    priority_score = compute_priority_score_from_factors(factors, PRIORITY_WEIGHTS)
    urgency = calculate_urgency(priority_score)
    planned_date = datetime.utcnow() + timedelta(hours=area.sla_hours)

    # Coordenadas aleatorias dentro de Vitacura
    lat, lon = random_vitacura_coords()

    # Asignación automática de cuadrilla (load balancing)
    squad = assign_squad_to_ticket(area_name, db)

    new_ticket = Ticket(
        title=ticket.title,
        description=ticket.description,
        priority_score=priority_score,
        urgency_level=urgency,
        status="Recibido",
        planned_date=planned_date,
        area_id=area.id,
        user_id=current_user.id,
        metrics_json=json.dumps(factors),
        priority_weights=json.dumps(PRIORITY_WEIGHTS),
        lat=lat,
        lon=lon,
        squad_name=squad,
    )

    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)

    # ─── Evidencia (máx 1 por ticket) ─────────────────────────────────────────
    evidence_id = None
    if ticket.image_url:
        existing_count = db.query(Evidence).filter(Evidence.ticket_id == new_ticket.id).count()
        if existing_count >= 1:
            raise HTTPException(status_code=400, detail="Este ticket ya tiene una foto asociada")

        ev = Evidence(
            ticket_id=new_ticket.id,
            image_url=ticket.image_url,
            description=(ticket.image_description or "")
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
        evidence_id = ev.id

    return {
        "ticket_id": new_ticket.id,
        "area": area.name,
        "priority": priority_score,
        "urgency_level": urgency,
        "planned_date": planned_date,
        "evidence_id": evidence_id,
        "squad_name": squad,
        "lat": lat,
        "lon": lon,
    }

@app.get("/my-tickets")
def my_tickets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tickets = db.query(Ticket).filter(Ticket.user_id == current_user.id).all()

    result = []
    for ticket in tickets:
        area = db.query(Area).filter(Area.id == ticket.area_id).first()
        assigned_user = db.query(User).filter(User.id == ticket.assigned_to).first() if ticket.assigned_to else None
        evidences = db.query(Evidence).filter(Evidence.ticket_id == ticket.id).all()

        result.append({
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
            "lon": ticket.lon,
            "evidences": [
                {
                    "image_url": ev.image_url,
                    "description": getattr(ev, "description", ""),
                    "created_at": ev.created_at
                }
                for ev in evidences
            ]
        })

    return result

@app.get("/tickets")
def get_tickets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden acceder")

    tickets = db.query(Ticket).order_by(Ticket.priority_score.desc()).all()

    result = []
    for ticket in tickets:
        area = db.query(Area).filter(Area.id == ticket.area_id).first()
        assigned_user = db.query(User).filter(User.id == ticket.assigned_to).first() if ticket.assigned_to else None
        reporter = db.query(User).filter(User.id == ticket.user_id).first()
        evidences = db.query(Evidence).filter(Evidence.ticket_id == ticket.id).all()

        result.append({
            "id": ticket.id,
            "title": ticket.title,
            "description": ticket.description,
            "status": ticket.status,
            "urgency_level": ticket.urgency_level,
            "priority_score": ticket.priority_score,
            "area_name": area.name if area else "Sin asignar",
            "squad_name": ticket.squad_name,
            "assigned_to": assigned_user.name if assigned_user else None,
            "reported_by": reporter.name if reporter else None,
            "reported_by_email": reporter.email if reporter else None,
            "created_at": ticket.created_at,
            "planned_date": ticket.planned_date,
            "lat": ticket.lat,
            "lon": ticket.lon,
            "evidences": [
                {
                    "image_url": ev.image_url,
                    "description": getattr(ev, "description", ""),
                    "created_at": ev.created_at
                }
                for ev in evidences
            ],
        })

    return result

class UpdateStatusRequest(BaseModel):
    status: str

@app.patch("/tickets/{ticket_id}/status")
def update_status(
    ticket_id: int,
    request: UpdateStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    prev_status = ticket.status
    ticket.status = request.status
    db.commit()

    # Liberar cuadrilla cuando el ticket se resuelve o cierra
    if request.status in ("Resuelto", "Cerrado") and prev_status not in ("Resuelto", "Cerrado"):
        release_squad_task(ticket.squad_name, db)

    return {"message": "Status updated", "new_status": request.status}


class AssignSquadRequest(BaseModel):
    squad_name: str

@app.patch("/tickets/{ticket_id}/assign")
def assign_ticket(
    ticket_id: int,
    request: AssignSquadRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden asignar tickets")

    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Ajustar contadores: liberar cuadrilla anterior, incrementar nueva
    if ticket.squad_name and ticket.squad_name != request.squad_name:
        release_squad_task(ticket.squad_name, db)
        new_squad = db.query(Squad).filter(Squad.name == request.squad_name).first()
        if new_squad:
            new_squad.pending_tasks += 1
            db.commit()

    ticket.squad_name = request.squad_name
    if ticket.status == "Recibido":
        ticket.status = "Asignado"
    db.commit()

    return {"message": "Squad assigned", "squad_name": request.squad_name}


class AddEvidenceRequest(BaseModel):
    image_url: str
    description: str = ""

@app.post("/tickets/{ticket_id}/evidence")
def add_evidence(
    ticket_id: int,
    request: AddEvidenceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
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
        description=request.description
    )
    db.add(evidence)
    db.commit()
    return {"message": "Evidence added", "evidence_id": evidence.id}


@app.get("/squads")
def get_squads(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Devuelve todas las cuadrillas con su contador de tareas pendientes."""
    if current_user.role not in ["operador", "operator", "supervisor"]:
        raise HTTPException(status_code=403, detail="Solo operadores pueden ver cuadrillas")

    squads = db.query(Squad).order_by(Squad.area_name, Squad.name).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "area_name": s.area_name,
            "pending_tasks": s.pending_tasks,
        }
        for s in squads
    ]


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
