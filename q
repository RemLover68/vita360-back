[33mcommit 5a0e6766db79f5584001d54c173df57e38f6acf7[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: RemLover68 <jose.delossantosmu@gmail.com>
Date:   Tue Mar 3 10:36:42 2026 -0300

    corrección de tickets no mostrándose, filtros mapa y frontend bonito

[1mdiff --git a/main.py b/main.py[m
[1mindex 213ef15..4eb627b 100644[m
[1m--- a/main.py[m
[1m+++ b/main.py[m
[36m@@ -13,6 +13,7 @@[m [mimport bcrypt[m
 import asyncio[m
 import httpx[m
 import json[m
[32m+[m[32mimport random[m
 import simulation_engine as sim[m
 [m
 # CONFIG[m
[36m@@ -345,6 +346,59 @@[m [mdef calculate_priority_factors_with_ai(title: str, description: str) -> dict:[m
     return factors[m
 [m
 [m
[32m+[m[32m# ─── Vitacura polygon helpers ─────────────────────────────────────────────────[m
[32m+[m
[32m+[m[32m# Polígono de la comuna de Vitacura (lon, lat)[m
[32m+[m[32mVITACURA_POLYGON = [[m
[32m+[m[32m    (-70.6061611, -33.4102650),[m
[32m+[m[32m    (-70.6041870, -33.4034583),[m
[32m+[m[32m    (-70.6041870, -33.3957911),[m
[32m+[m[32m    (-70.5981789, -33.3894849),[m
[32m+[m[32m    (-70.5933723, -33.3851849),[m
[32m+[m[32m    (-70.5849609, -33.3812431),[m
[32m+[m[32m    (-70.5748329, -33.3794513),[m
[32m+[m[32m    (-70.5653229, -33.3770144),[m
[32m+[m[32m    (-70.5573406, -33.3758676),[m
[32m+[m[32m    (-70.5485001, -33.3742907),[m
[32m+[m[32m    (-70.5423203, -33.3756500),[m
[32m+[m[32m    (-70.5380249, -33.3807000),[m
[32m+[m[32m    (-70.5360000, -33.3900000),[m
[32m+[m[32m    (-70.5390000, -33.4050000),[m
[32m+[m[32m    (-70.5500000, -33.4150000),[m
[32m+[m[32m    (-70.5650000, -33.4200000),[m
[32m+[m[32m    (-70.5850000, -33.4200000),[m
[32m+[m[32m    (-70.6000000, -33.4160000),[m
[32m+[m[32m    (-70.6061611, -33.4102650),[m
[32m+[m[32m][m
[32m+[m
[32m+[m[32mdef _point_in_polygon(x: float, y: float, poly: list) -> bool:[m
[32m+[m[32m    """Ray-casting algorithm: returns True if point (x,y) is inside polygon."""[m
[32m+[m[32m    n = len(poly)[m
[32m+[m[32m    inside = False[m
[32m+[m[32m    j = n - 1[m
[32m+[m[32m    for i in range(n):[m
[32m+[m[32m        xi, yi = poly[i][m
[32m+[m[32m        xj, yj = poly[j][m
[32m+[m[32m        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):[m
[32m+[m[32m            inside = not inside[m
[32m+[m[32m        j = i[m
[32m+[m[32m    return inside[m
[32m+[m
[32m+[m[32mdef _random_point_in_vitacura() -> tuple:[m
[32m+[m[32m    """Return a random (lat, lng) strictly inside the Vitacura polygon."""[m
[32m+[m[32m    lons = [p[0] for p in VITACURA_POLYGON][m
[32m+[m[32m    lats = [p[1] for p in VITACURA_POLYGON][m
[32m+[m[32m    min_lon, max_lon = min(lons), max(lons)[m
[32m+[m[32m    min_lat, max_lat = min(lats), max(lats)[m
[32m+[m[32m    for _ in range(1000):[m
[32m+[m[32m        lon = random.uniform(min_lon, max_lon)[m
[32m+[m[32m        lat = random.uniform(min_lat, max_lat)[m
[32m+[m[32m        if _point_in_polygon(lon, lat, VITACURA_POLYGON):[m
[32m+[m[32m            return lat, lon[m
[32m+[m[32m    # Fallback: centroid of Vitacura if somehow never lands inside[m
[32m+[m[32m    return -33.3947, -70.5680[m
[32m+[m
[32m+[m
 def compute_priority_score_from_factors(factors: dict, weights: dict) -> int:[m
     total = 0.0[m
     for key, weight in weights.items():[m
[36m@@ -478,6 +532,12 @@[m [mdef create_ticket([m
     urgency = calculate_urgency(priority_score)[m
     planned_date = datetime.utcnow() + timedelta(hours=area.sla_hours)[m
 [m
[32m+[m[32m    # Usar coordenadas del ciudadano o generar punto aleatorio dentro de Vitacura[m
[32m+[m[32m    if ticket.lat is not None and ticket.lng is not None:[m
[32m+[m[32m        ticket_lat, ticket_lng = ticket.lat, ticket.lng[m
[32m+[m[32m    else:[m
[32m+[m[32m        ticket_lat, ticket_lng = _random_point_in_vitacura()[m
[32m+[m
     new_ticket = Ticket([m
         title=ticket.title,[m
         description=ticket.description,[m
[36m@@ -487,8 +547,8 @@[m [mdef create_ticket([m
         planned_date=planned_date,[m
         area_id=area.id,[m
         user_id=current_user.id,[m
[31m-        lat=ticket.lat,[m
[31m-        lng=ticket.lng,[m
[32m+[m[32m        lat=ticket_lat,[m
[32m+[m[32m        lng=ticket_lng,[m
         metrics_json=json.dumps(factors),[m
         priority_weights=json.dumps(PRIORITY_WEIGHTS),[m
     )[m
[36m@@ -539,17 +599,38 @@[m [mdef my_tickets(current_user: User = Depends(get_current_user), db: Session = Dep[m
     tickets = db.query(Ticket).filter(Ticket.user_id == current_user.id).all()[m
     return [_serialize_ticket(t, db) for t in tickets][m
 [m
[32m+[m[32m@app.get("/tickets/count")[m
[32m+[m[32mdef get_tickets_count([m
[32m+[m[32m    current_user: User = Depends(get_current_user),[m
[32m+[m[32m    db: Session = Depends(get_db),[m
[32m+[m[32m):[m
[32m+[m[32m    """Endpoint ligero para el monitor de IA.[m
[32m+[m[32m    Devuelve solo el total de tickets sin serializar nada.[m
[32m+[m[32m    Solo accesible por operadores/supervisores."""[m
[32m+[m[32m    if current_user.role not in ["operador", "operator", "supervisor"]:[m
[32m+[m[32m        raise HTTPException(status_code=403, detail="Solo operadores pueden acceder")[m
[32m+[m[32m    count = db.query(Ticket).count()[m
[32m+[m[32m    return {"count": count}[m
[32m+[m
 @app.get("/tickets")[m
 def get_tickets([m
     status: Optional[str] = None,[m
     area: Optional[str] = None,[m
[32m+[m[32m    limit: Optional[int] = None,[m
[32m+[m[32m    offset: Optional[int] = 0,[m
[32m+[m[32m    order: Optional[str] = "desc",[m
     current_user: User = Depends(get_current_user),[m
     db: Session = Depends(get_db),[m
 ):[m
     if current_user.role not in ["operador", "operator", "supervisor"]:[m
         raise HTTPException(status_code=403, detail="Solo operadores pueden acceder")[m
 [m
[31m-    query = db.query(Ticket).order_by(Ticket.priority_score.desc())[m
[32m+[m[32m    query = db.query(Ticket)[m
[32m+[m[32m    if order == "asc":[m
[32m+[m[32m        query = query.order_by(Ticket.id.asc())[m
[32m+[m[32m    else:[m
[32m+[m[32m        query = query.order_by(Ticket.priority_score.desc(), Ticket.id.desc())[m
[32m+[m
     if status:[m
         query = query.filter(Ticket.status == status)[m
     if area:[m
[36m@@ -557,6 +638,11 @@[m [mdef get_tickets([m
         if area_obj:[m
             query = query.filter(Ticket.area_id == area_obj.id)[m
 [m
[32m+[m[32m    if offset:[m
[32m+[m[32m        query = query.offset(offset)[m
[32m+[m[32m    if limit:[m
[32m+[m[32m        query = query.limit(limit)[m
[32m+[m
     tickets = query.all()[m
     return [_serialize_ticket(t, db, include_reporter=True) for t in tickets][m
 [m
