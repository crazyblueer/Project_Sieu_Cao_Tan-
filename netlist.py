import numpy as np
import os
from components import (
    to_si, resistor_series, resistor_shunt,
    capacitor_series, capacitor_shunt,
    inductor_series, inductor_shunt,
    tline_general, beta_l_from_freq,
    junction_t, open_circuit, short_circuit
)
from touchstone import load_s2p


class CircuitNetlist:
    def __init__(self):
        self.freqs = np.array([])         
        self.raw_lines = []
        self.external_ports = {}          
        self.elements = []               
        self.node_connections = {}       
        
    def parse_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('#') or line_str.startswith('!'):
                    continue
                self._parse_line(line_str)
        return self

    def parse_text(self, text_content):
        for line in text_content.strip().split('\n'):
            line_str = line.strip()
            if not line_str or line_str.startswith('#') or line_str.startswith('!'):
                continue
            self._parse_line(line_str)
        return self

    def _parse_line(self, line):
        tokens = line.split()
        cmd = tokens[0].upper()

        
        if cmd == "FREQ":
            params = {}
            for t in tokens[1:]:
                k, v = t.split('=')
                params[k.upper()] = v
            f_start = to_si(params.get('START', '1G'))
            f_stop = to_si(params.get('STOP', '3G'))
            points = int(params.get('POINTS', 101))
            self.freqs = np.linspace(f_start, f_stop, points)

        
        elif cmd == "PORT":
            port_num = int(tokens[1])
            node_name = tokens[2]
            self.external_ports[port_num] = node_name

        
        else:
            elem_type = cmd
            name = tokens[1]
            params = {}
            nodes = []
            for t in tokens[2:]:
                if '=' in t:
                    k, v = t.split('=')
                    params[k.upper()] = v
                else:
                    nodes.append(t)
            self.elements.append({
                'type': elem_type,
                'name': name,
                'nodes': nodes,
                'params': params
            })

    def build_network_at_freq(self, f_val, z0=50.0):
        
        blocks = []        
        connections = []   
        node_pins = {}     

        def add_pin(node, pin_label):
            node_pins.setdefault(node, []).append(pin_label)

        
        for elem in self.elements:
            t = elem['type']
            name = elem['name']
            nodes = elem['nodes']
            p = elem['params']

            if t == "RES":
                r_val = to_si(p.get('R', 50))
                pin_in, pin_out = f"{name}_p1", f"{name}_p2"
                blocks.append((resistor_series(r_val, z0), [pin_in, pin_out]))
                add_pin(nodes[0], pin_in)
                add_pin(nodes[1], pin_out)

            elif t == "TLINE":
                z_c = to_si(p.get('Z0', 50))
                el_deg = float(p.get('EL', 90))
                f0 = to_si(p.get('F0', f_val))
                theta = beta_l_from_freq(f_val, f0=f0, electrical_length_deg=el_deg)
                pin_in, pin_out = f"{name}_p1", f"{name}_p2"
                blocks.append((tline_general(theta, z_c, z0), [pin_in, pin_out]))
                add_pin(nodes[0], pin_in)
                add_pin(nodes[1], pin_out)

            elif t == "CAP":
                c_val = to_si(p.get('C', '1p'))
                pin_in, pin_out = f"{name}_p1", f"{name}_p2"
                blocks.append((capacitor_series(c_val, f=f_val, Z0=z0), [pin_in, pin_out]))
                add_pin(nodes[0], pin_in)
                add_pin(nodes[1], pin_out)

            elif t == "IND":
                l_val = to_si(p.get('L', '1n'))
                pin_in, pin_out = f"{name}_p1", f"{name}_p2"
                blocks.append((inductor_series(l_val, f=f_val, Z0=z0), [pin_in, pin_out]))
                add_pin(nodes[0], pin_in)
                add_pin(nodes[1], pin_out)

            elif t == "S2P":
                filename = p.get('FILE')
                ts = load_s2p(filename)
                s_mat = ts.s_interp(f_val)
                pin_in, pin_out = f"{name}_p1", f"{name}_p2"
                blocks.append((s_mat, [pin_in, pin_out]))
                add_pin(nodes[0], pin_in)
                add_pin(nodes[1], pin_out)

        
        ext_port_labels = []
        ext_ports_sorted = sorted(self.external_ports.items(), key=lambda x: x[0])

        junc_count = 0
        for port_num, node in ext_ports_sorted:
            
            ext_label = f"EXT_P{port_num}"
            ext_port_labels.append(ext_label)
            add_pin(node, ext_label)

        
        for node, pins in node_pins.items():
            if len(pins) == 2:
                
                connections.append((pins[0], pins[1]))
            elif len(pins) == 3:
                
                junc_count += 1
                j_name = f"J_{junc_count}"
                j_pins = [f"{j_name}_1", f"{j_name}_2", f"{j_name}_3"]
                blocks.append((junction_t(), j_pins))
                
                connections.append((pins[0], j_pins[0]))
                connections.append((pins[1], j_pins[1]))
                connections.append((pins[2], j_pins[2]))
            elif len(pins) > 3:
                raise NotImplementedError(f"Node {node} has {len(pins)} pins (requires a connection expander for > 3 ports).")

        return blocks, connections, ext_port_labels


def read_netlist(filename):
    
    net = CircuitNetlist()
    net.parse_file(filename)
    return net