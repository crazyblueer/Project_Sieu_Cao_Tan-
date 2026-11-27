"""Topic 7 S-Matrix Circuit Editor v4 — generic topology-aware layout.

Drop this file + schematic_layout.py beside circuit_gui_v3.py and the team's
netlist.py, solver.py, components.py, touchstone.py.

No changes to the solver or the netlist's electrical connections are needed.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from circuit_gui_v3 import SchematicAppV3, detect_wilkinson, COLORS
from schematic_layout import generate_layout, is_ground


class SchematicAppV4(SchematicAppV3):
    """V3 simulation/optimization + generic schematic layout with panning."""

    def _init_ui(self):
        self._generic = None
        self._show_node_names = tk.BooleanVar(value=False)
        super()._init_ui()
        self.root.title('Topic 7 | S-Matrix Circuit Editor v4 · Generic Layout')
        diagram = self.canvas.master
        self.canvas.pack_forget()
        vertical = ttk.Scrollbar(diagram, orient='vertical', command=self.canvas.yview)
        horizontal = ttk.Scrollbar(diagram, orient='horizontal', command=self.canvas.xview)
        vertical.pack(side='right', fill='y')
        horizontal.pack(side='bottom', fill='x')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.configure(xscrollcommand=horizontal.set,
                              yscrollcommand=vertical.set,
                              scrollregion=(0,0,1100,640))
        # Middle click/drag pans a large diagram; Shift+wheel scrolls sideways.
        self.canvas.bind('<ButtonPress-2>', lambda e: self.canvas.scan_mark(e.x,e.y))
        self.canvas.bind('<B2-Motion>', lambda e: self.canvas.scan_dragto(e.x,e.y,gain=1))
        self.canvas.bind('<Shift-MouseWheel>',
                         lambda e: self.canvas.xview_scroll(-int(e.delta/120), 'units'))
        self.canvas.bind('<MouseWheel>',
                         lambda e: self.canvas.yview_scroll(-int(e.delta/120), 'units'))
        self.canvas.bind('<Button-4>', lambda e: self.canvas.yview_scroll(-3, 'units'))
        self.canvas.bind('<Button-5>', lambda e: self.canvas.yview_scroll(3, 'units'))
        self.canvas.bind('<Configure>', lambda e: self.redraw())

        toolbar = self.root.winfo_children()[0]
        ttk.Button(toolbar, text='Reset view', command=self.fit_view).pack(side='left',padx=(8,2))
        ttk.Checkbutton(toolbar, text='Node names', variable=self._show_node_names,
                        command=self.redraw).pack(side='left',padx=2)
        self.status.set('v4: generic connectivity-based layout | scroll or middle-drag to pan')

    def fit_view(self):
        # Reset scroll to the circuit's upper-left; large circuits remain scrollable.
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def open_path(self, path):
        self._generic=None
        super().open_path(path)
        self.fit_view()

    def new_document(self):
        self._generic=None
        super().new_document()

    def auto_layout(self):
        """Pure layout transformation: never changes electrical connectivity."""
        if detect_wilkinson(self.elements,self.ports) is not None:
            self._generic=None
            super().auto_layout()  # Keep the already-validated ideal Wilkinson drawing.
            self.canvas.configure(scrollregion=(0,0,1200,600))
            self.fit_view()
            return
        self._wilkinson=None
        self._generic=generate_layout(self.elements,self.ports)
        self.positions=dict(self._generic.positions)
        self.active_pin=None
        self.canvas.configure(scrollregion=(0,0,self._generic.width,self._generic.height))
        extra=(' | '+self._generic.notes[0]) if self._generic.notes else ''
        self.status.set('Generic topology-based layout complete · electrical nodes unchanged'+extra)
        self.redraw()
        self.fit_view()

    def add_port(self):
        self._generic=None
        super().add_port()
        self.auto_layout()

    def add_element(self, kind):
        self._generic=None
        super().add_element(kind)
        self.auto_layout()

    def edit_selected(self):
        self._generic=None
        super().edit_selected()
        self.auto_layout()

    def delete_selected(self):
        self._generic=None
        super().delete_selected()
        self.auto_layout()

    def _merge_nodes(self, source, destination):
        self._generic=None
        super()._merge_nodes(source,destination)
        # During mouse events redraw occurs immediately afterwards; create new
        # display coordinates before drawing this changed electrical graph.
        if self._wilkinson is None:
            self._generic=generate_layout(self.elements,self.ports)
            self.positions=dict(self._generic.positions)

    def _geometry(self, el):
        if self._generic is None:
            return super()._geometry(el)
        x,y=self._center(('el',el['name']))
        kind=self._generic.orientations.get(el['name'],'h')
        count=len(el['nodes'])
        if kind == 'four' and count==4:
            return False, [(x-72,y-19),(x+72,y-19),
                           (x-72,y+19),(x+72,y+19)]
        if kind in ('one','shunt'):
            live=[n for n in el['nodes'] if not is_ground(n)]
            anchor=self._generic.node_positions.get(live[0],(x,y-130)) if live else (x,y-130)
            above = y < anchor[1]
            if count==1:
                return True,[(x,y+35) if above else (x,y-35)]
            live_pin=(x,y+35) if above else (x,y-35)
            ground_pin=(x,y-35) if above else (x,y+35)
            return True,[ground_pin if is_ground(n) else live_pin
                         for n in el['nodes']]
        if kind=='v' and count==2:
            n1,n2=el['nodes']
            ay=self._generic.node_positions.get(n1,(x,y-100))[1]
            by=self._generic.node_positions.get(n2,(x,y+100))[1]
            return True, ([(x,y-37),(x,y+37)] if ay<=by
                          else [(x,y+37),(x,y-37)])
        if count==2:
            a,b=el['nodes']
            ax=self._generic.node_positions.get(a,(x-100,y))[0]
            bx=self._generic.node_positions.get(b,(x+100,y))[0]
            return False, ([(x-70,y),(x+70,y)] if ax<=bx
                           else [(x+70,y),(x-70,y)])
        if count==1:
            return True,[(x,y-35)]
        return super()._geometry(el)

    def _port_pin(self,p):
        x,y=self._center(('port',p))
        node=self.ports[p]['node']
        hub=self._generic.node_positions.get(node,(x+90,y))
        return (x+44 if hub[0]>=x else x-44,y)

    def redraw(self):
        if not hasattr(self,'canvas'):
            return
        if self._generic is None:
            return super().redraw()
        cv=self.canvas
        cv.delete('all')
        self.pin_hit=[]
        self.pin_coords={}
        colors={'line':'#64748b','junction':'#334155'}
        # Compute terminal coordinates first; every connection is made by shared
        # node name, never by a visual line crossing.
        for el in self.elements:
            key=('el',el['name'])
            _,points=self._geometry(el)
            for i,(px,py) in enumerate(points):
                self._pin((key,i),el['nodes'][i],px,py)
        for p,data in self.ports.items():
            px,py=self._port_pin(p)
            self._pin((('port',p),0),data['node'],px,py)
        members={}
        for pin,(node,x,y) in self.pin_coords.items():
            if not is_ground(node):
                members.setdefault(node,[]).append((x,y))

        # Bus junctions make multi-terminal nodes visually explicit. Routes are
        # orthogonal; wires are drawn below device blocks and pin handles.
        for node,pts in members.items():
            hub=self._generic.node_positions.get(node)
            if hub is None:
                if len(pts)<2:
                    continue
                hub=(sum(x for x,_ in pts)/len(pts),sum(y for _,y in pts)/len(pts))
            hx,hy=hub
            for x,y in pts:
                if abs(x-hx)<0.01 and abs(y-hy)<0.01:
                    continue
                # Choose an elbow leaving the pin parallel to its terminal.
                if abs(x-hx)<0.01 or abs(y-hy)<0.01:
                    cv.create_line(x,y,hx,hy,fill=colors['line'],width=2)
                else:
                    # If pin lies above/below net, first route horizontally to
                    # the node's x bus; otherwise turn towards its y bus.
                    cv.create_line(x,y,hx,y,hx,hy,fill=colors['line'],width=2,
                                   joinstyle='round')
            if len(pts)>=3:
                cv.create_oval(hx-4,hy-4,hx+4,hy+4,
                               fill=colors['junction'],outline='')
            if self._show_node_names.get():
                cv.create_text(hx+5,hy-11,text=str(node),anchor='w',
                               fill='#52647b',font=('Helvetica',9))

        # Components and their stubs sit ON TOP of wires, so crossing without
        # electrical node identity never becomes a connection.
        for el in self.elements:
            key=('el',el['name']);x,y=self._center(key)
            kind=self._generic.orientations.get(el['name'],'h')
            color=COLORS.get(el['type'],'#475569')
            outline='#f97316' if self.selected==key else color
            _,pts=self._geometry(el)
            if kind=='four':
                halfw,halfh=59,42
            elif kind in ('one','shunt','v'):
                halfw,halfh=45,27
            else:
                halfw,halfh=57,26
            for xx,yy in pts:
                # Extend each pin to the closest boundary of the device body.
                if xx < x-halfw:
                    target=(x-halfw,yy)
                elif xx > x+halfw:
                    target=(x+halfw,yy)
                elif yy<y-halfh:
                    target=(xx,y-halfh)
                else:
                    target=(xx,y+halfh)
                cv.create_line(xx,yy,*target,fill=color,width=2)
            cv.create_rectangle(x-halfw,y-halfh,x+halfw,y+halfh,
                                fill='white',outline=outline,width=2)
            cv.create_text(x,y-8,text=f"{el['type']} {el['name']}",
                           fill='#0f172a',font=('Helvetica',10,'bold'))
            if el['type']=='S2P':
                info=el['params'].get('FILE','S2P data')
            elif el['type']=='STUB':
                info=(f"{el['params'].get('Z0','?')}Ω "
                      f"{el['params'].get('EL','?')}° {el['params'].get('TERM','open')}")
            elif el['type']=='TLINE':
                raw_z=el['params'].get('Z0','?')
                try:
                    z_text=f'{float(raw_z):.5g}'
                except (TypeError,ValueError):
                    z_text=str(raw_z)
                info=f"Z0={z_text} EL={el['params'].get('EL','?')}°"
            else:
                info=' '.join(f'{k}={v}' for k,v in el['params'].items())
            limit=18 if kind in ('v','one','shunt') else 24
            cv.create_text(x,y+11,text=info[:limit]+('…' if len(info)>limit else ''),
                           fill='#64748b',font=('Helvetica',9))

        for p in sorted(self.ports):
            key=('port',p);x,y=self._center(key)
            outline='#f97316' if self.selected==key else '#2563eb'
            cv.create_oval(x-29,y-21,x+29,y+21,
                           fill='#dbeafe',outline=outline,width=2)
            cv.create_text(x,y,text=f'P{p}',fill='#1e40af',
                           font=('Helvetica',11,'bold'))

        # Draw ground symbols independently (all GND nodes are equivalent,
        # but a physically crossing line must not be misread as a junction).
        for pin,(node,x,y) in self.pin_coords.items():
            if is_ground(node):
                cv.create_line(x-10,y+1,x+10,y+1,fill='#059669',width=2)
                cv.create_line(x-7,y+5,x+7,y+5,fill='#059669',width=2)
                cv.create_line(x-4,y+9,x+4,y+9,fill='#059669',width=2)
            color=('#ef4444' if self.active_pin==pin else
                   '#059669' if is_ground(node) else '#111827')
            cv.create_oval(x-5,y-5,x+5,y+5,fill=color,outline='white',width=1.2)
        cv.create_text(16,15,anchor='w',
                       text='GENERIC · connectivity-aware layout · wires are node-name based',
                       fill='#64748b',font=('Helvetica',10,'bold'))
        cv.configure(scrollregion=(0,0,self._generic.width,self._generic.height))

    def _canvas_event(self,event):
        # Tk event coords are viewport-relative; convert to scroll-canvas coords.
        class Event:
            pass
        out=Event()
        out.x=self.canvas.canvasx(event.x)
        out.y=self.canvas.canvasy(event.y)
        return out

    def mouse_down(self,event):
        return super().mouse_down(self._canvas_event(event))

    def mouse_move(self,event):
        return super().mouse_move(self._canvas_event(event))

    def double_click(self,event):
        return super().double_click(self._canvas_event(event))


if __name__ == '__main__':
    root=tk.Tk()
    app=SchematicAppV4(root)
    root.mainloop()
