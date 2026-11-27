"""Deterministic, dependency-free connectivity-based schematic layout.

Pure presentation layer. It reads the GUI's parsed elements/ports and returns
positions and terminal orientations; it NEVER changes node names or topology.

Design: shortest weighted main signal path (active devices preferred over
feedback resistors), lower lanes for auxiliary branches/ports, upper lanes for
chords/feedback, vertical ground/stub symbols, Manhattan node-bus routing.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import itertools

GROUND = {'0', 'GND'}


def is_ground(node):
    return str(node).upper() in GROUND


@dataclass
class Layout:
    positions: dict                  # ('el', name) / ('port', number) -> (x,y)
    node_positions: dict             # net name -> (x,y), routing anchors
    orientations: dict               # element name -> 'h','v','shunt','one','four'
    width: int
    height: int
    notes: tuple = ()


def _edge_weight(el):
    typ = el['type']
    if typ == 'S2P':
        return 0.55  # prefer active signal path, not a parallel feedback resistor
    if typ == 'RES':
        return 3.2
    if typ in ('TLINE', 'MLIN', 'IND', 'CAP', 'CLIN'):
        return 1.0
    return 1.5


def _shortest(start, goals, adjacency):
    """Return [(u, v, element), ...] from start to any goal, or None."""
    if start in goals:
        return []
    counter = itertools.count()
    todo = [(0.0, 0, next(counter), start)]
    best = {start: (0.0, 0)}
    prev = {}
    while todo:
        cost, hops, _, node = heapq.heappop(todo)
        if (cost, hops) != best.get(node):
            continue
        if node in goals:
            path = []
            while node != start:
                before, name = prev[node]
                path.append((before, node, name))
                node = before
            return list(reversed(path))
        for other, name, weight in sorted(adjacency[node], key=lambda row: (row[2], row[1])):
            cand = (cost+weight, hops+1)
            if other not in best or cand < best[other]:
                best[other] = cand
                prev[other] = (node, name)
                heapq.heappush(todo, (*cand, next(counter), other))
    return None


def generate_layout(elements, ports):
    """Lay out any valid supported netlist, including parallel paths and cycles.

    For >2-terminal CLIN blocks, approximate their internal topology for the
    *layout search only*. The full original terminals remain separate in GUI.
    """
    elements = list(elements)
    edges = {}
    adj = defaultdict(list)
    for el in elements:
        name = el['name']
        live = [n for n in el['nodes'] if not is_ground(n)]
        if len(el['nodes']) == 2 and len(live) == 2 and live[0] != live[1]:
            a, b = live
            edges[name] = (a, b)
            weight = _edge_weight(el)
            adj[a].append((b, name, weight))
            adj[b].append((a, name, weight))
        elif len(el['nodes']) == 4 and len(live) == 4:
            # CLIN is represented as one four-terminal component. These
            # virtual links affect only search; not simulator connectivity.
            for j, (a,b) in enumerate(((live[0],live[1]), (live[2],live[3]))):
                tag = f'{name}#virtual{j}'
                adj[a].append((b,tag,1.2))
                adj[b].append((a,tag,1.2))

    by_name = {el['name']:el for el in elements}
    keys = sorted(ports)
    pnodes = {p:ports[p]['node'] for p in keys}
    pos = {}
    nodes = {}
    orientation = {}
    notes = []
    used = set()
    main_y = 185 if len(keys)>2 else 215
    x0, dx = 210, 215

    # Main signal path from the first to second port, weighted to prefer
    # transistor-based transmission rather than bypass feedback.
    path = None
    if len(keys) >= 2:
        path = _shortest(pnodes[keys[0]], {pnodes[keys[1]]}, adj)
    if path:
        chain = [path[0][0]] + [b for _,b,_ in path]
        for i, node in enumerate(chain):
            nodes[node] = (x0 + i*dx, main_y)
        for a,b,name in path:
            if name in by_name:
                x1,y1 = nodes[a]; x2,y2 = nodes[b]
                pos[('el',name)] = ((x1+x2)/2,(y1+y2)/2)
                orientation[name] = 'h'
                used.add(name)
    else:
        if keys:
            nodes[pnodes[keys[0]]] = (x0,main_y)
        if len(keys) >= 2 and pnodes[keys[1]] not in nodes:
            nodes[pnodes[keys[1]]] = (x0+dx,main_y)
        notes.append('No two-terminal path between first two external ports; disconnected regions shown separately')

    main_nodes = set(nodes)
    # Additional ports form secondary horizontal/vertical lanes. For equal
    # branch lengths from different main anchors, keep them on the same lane
    # (e.g. a branch-line hybrid's lower edge).
    allocated = defaultdict(int)
    for p in keys[2:]:
        target = pnodes[p]
        if target in nodes:
            continue
        # Prefer a direct attachment to an original main-path node on ties.
        # This keeps branch-line hybrids rectangular instead of collapsing
        # two auxiliary output ports onto the same branch coordinate.
        main_route = _shortest(target, main_nodes, adj)
        any_route = _shortest(target, set(nodes), adj)
        def route_cost(path):
            if path is None:
                return float('inf')
            result=0.0
            for u,v,name in path:
                match=next((w for other,label,w in adj[u]
                            if other==v and label==name),1.0)
                result+=match
            return result
        route = (main_route if main_route is not None and
                 route_cost(main_route)<=route_cost(any_route)+0.01
                 else any_route)
        if route is None:
            continue
        route = [(b,a,name) for a,b,name in reversed(route)]  # placed anchor -> target
        anchor = route[0][0]
        ax,ay = nodes[anchor]
        depth = allocated[anchor]
        allocated[anchor] += 1
        # Put first branch below main. Additional branches at same anchor are
        # staggered instead of overlapping one another.
        lane_y = main_y + 145 + depth*160
        for j,(a,b,name) in enumerate(route):
            if b not in nodes:
                if j == 0:
                    bx = ax if ax >= x0+dx-30 else ax
                    by = lane_y
                else:
                    bx = nodes[a][0]+dx
                    by = lane_y
                nodes[b]=(bx,by)
            x1,y1=nodes[a];x2,y2=nodes[b]
            if name in by_name and name not in used:
                pos[('el',name)]=((x1+x2)/2,(y1+y2)/2)
                orientation[name]='v' if abs(y2-y1)>abs(x2-x1) else 'h'
                used.add(name)

    # Place any remaining connected components. If an edge spans established
    # nodes, use it as a chord (feedback loop), keeping it away from main chain.
    for el in elements:
        name=el['name']
        if name in used or name not in edges:
            continue
        a,b=edges[name]
        if a in nodes and b in nodes:
            x1,y1=nodes[a];x2,y2=nodes[b]
            if abs(y2-y1) < 45 and abs(x2-x1)>120:
                mx=(x1+x2)/2
                # If there is an existing component on this span, move this
                # chord to an external feedback lane.
                occupied=any(abs(pxy[0]-mx)<100 and abs(pxy[1]-y1)<75
                             for k,pxy in pos.items() if k[0]=='el')
                cy = (min(y1,y2)-118) if occupied else (y1+y2)/2
                pos[('el',name)]=(mx,cy)
                orientation[name]='h'
            elif abs(x2-x1)<70 and abs(y2-y1)>65:
                pos[('el',name)]=((x1+x2)/2,(y1+y2)/2)
                orientation[name]='v'
            else:
                # General cycle/chord: orthogonal trunk lane outside chain.
                pos[('el',name)]=((x1+x2)/2,min(y1,y2)-135)
                orientation[name]='h'
            used.add(name)

    # Grow remaining non-port branches from a known node. Handles incomplete,
    # complex and disconnected networks without pretending the wires connect.
    pending=[el for el in elements if el['name'] in edges and el['name'] not in used]
    nstep = 0
    while pending and nstep < len(elements)+1:
        nstep += 1
        progress=False
        for el in list(pending):
            a,b=edges[el['name']]
            if (a in nodes) ^ (b in nodes):
                start,end = (a,b) if a in nodes else (b,a)
                sx,sy=nodes[start]
                branch_index=sum(1 for name in used if start in edges.get(name,()))
                ty=sy+180+(branch_index%3)*75
                tx=sx+dx if branch_index%2==0 else sx-dx
                nodes[end]=(max(x0,tx),ty)
                pos[('el',el['name'])]=((nodes[start][0]+nodes[end][0])/2,
                                       (nodes[start][1]+nodes[end][1])/2)
                orientation[el['name']]='h' if abs(tx-sx)>abs(ty-sy) else 'v'
                used.add(el['name'])
                pending.remove(el)
                progress=True
        if not progress:
            break
    orphan_index=0
    for el in pending:
        a,b=edges[el['name']]
        x= x0 + (orphan_index%3)*dx
        y=main_y+420+(orphan_index//3)*170
        nodes.setdefault(a,(x-95,y))
        nodes.setdefault(b,(x+95,y))
        pos[('el',el['name'])]=(x,y)
        orientation[el['name']]='h'
        used.add(el['name'])
        orphan_index+=1

    # Four-terminal components: locate them beside their previously laid-out
    # nets; keep four distinct pins rather than inventing electrical joins.
    multi_index = 0
    for el in elements:
        if el['name'] in used or len(el['nodes'])<=2:
            continue
        match=[nodes[n] for n in el['nodes'] if n in nodes]
        if match:
            cx=sum(x for x,_ in match)/len(match)
            cy=sum(y for _,y in match)/len(match)+140
        else:
            cx=x0+multi_index*dx
            cy=main_y+420
        pos[('el',el['name'])]=(cx,cy)
        orientation[el['name']]='four'
        used.add(el['name'])
        multi_index+=1
        for i,n in enumerate(el['nodes']):
            if not is_ground(n):
                nodes.setdefault(n,(cx+(-105 if i%2==0 else 105),
                                    cy+(-50 if i<2 else 50)))

    # One-terminal STUB, OPEN/SHORT and grounded R/L/C are vertical branches.
    count_on_node=defaultdict(int)
    floating=0
    for el in elements:
        if el['name'] in used:
            continue
        live=[n for n in el['nodes'] if not is_ground(n)]
        if len(live)==1:
            node=live[0]
            if node not in nodes:
                nodes[node]=(x0+floating*dx,main_y+420)
                floating+=1
            nx,ny=nodes[node]
            index=count_on_node[node]
            count_on_node[node]+=1
            side = 1 if index%2==0 else -1
            offset=82*(index//2)
            cx=nx+offset
            cy=ny+side*113
            pos[('el',el['name'])]=(cx,cy)
            orientation[el['name']]='shunt' if len(el['nodes'])==2 else 'one'
        else:
            pos[('el',el['name'])]=(x0+floating*dx,main_y+420)
            orientation[el['name']]='h'
            floating+=1
        used.add(el['name'])

    # Put external ports outside the circuit on their respective sides.
    for i,p in enumerate(keys):
        node=pnodes[p]
        if node not in nodes:
            nodes[node]=(x0+(i%4)*dx, main_y+420+(i//4)*160)
        x,y=nodes[node]
        if p==keys[0]:
            px=x-100
        elif len(keys)>1 and p==keys[1]:
            px=x+100
        else:
            # Lower left/right ports follow the node's place relative to main.
            px=x-100 if x<=x0+20 else x+100
        pos[('port',p)]=(px,y)

    # If a vertical line shares a node with a ground shunt, it can look like
    # a series part; this orientation is intentional and visually explicit.
    min_x=min((x for x,y in pos.values()),default=0)
    min_y=min((y for x,y in pos.values()),default=0)
    if min_x<95 or min_y<65:
        shift_x=max(0,100-min_x)
        shift_y=max(0,90-min_y)
        pos={key:(x+shift_x,y+shift_y) for key,(x,y) in pos.items()}
        nodes={key:(x+shift_x,y+shift_y) for key,(x,y) in nodes.items()}
    max_x=max([x for x,y in pos.values()]+[950])
    max_y=max([y for x,y in pos.values()]+[450])
    return Layout(pos,nodes,orientation,int(max_x+160),int(max_y+110),tuple(notes))
