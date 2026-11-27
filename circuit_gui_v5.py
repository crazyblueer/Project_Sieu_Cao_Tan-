"""Topic 7 S-matrix Circuit Editor v5 — release integration.

Keeps v4 generic topology layout, v3 multi-port S plots and LC optimizer.
Adds: noise lab (passive multiports + explicitly modeled active S2P),
matched termination handling, noise CSV/plots, and 4-port branch-line report.
The team's RF solver/netlist/connection algorithms are UNMODIFIED.
"""
from __future__ import annotations

import csv
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import numpy as np

from circuit_gui_v4 import SchematicAppV4
from noise_gui_analysis import (evaluate_noise_at, evaluate_noise_sweep,
                                branchline_hybrid_check)


class SchematicAppV5(SchematicAppV4):
    """Final GUI integration; noise UI is an additional nonmodal lab window."""

    def __init__(self, root):
        self._noise_window = None
        self._noise_last = None
        self._noise_running = False
        self._noise_queue = queue.Queue()
        self._noise_token = 0
        super().__init__(root)

    def _init_ui(self):
        super()._init_ui()
        self.root.title('Topic 7 | S-Matrix Circuit Editor v5 · Simulation / Optimization / Noise')
        toolbar = next(child for child in self.root.winfo_children()
                       if isinstance(child, ttk.Frame))
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=7)
        ttk.Button(toolbar, text='Noise Lab', command=self.open_noise_lab).pack(side='left', padx=3)
        ttk.Button(toolbar, text='Check 4-port', command=self.check_four_port).pack(side='left', padx=3)
        self.status.set('v5 ready | S-matrix · generic layout · LC optimization · noise analysis')

    def _invalidate_simulation(self):
        super()._invalidate_simulation()
        self._noise_last = None
        self._noise_token += 1
        if self._noise_window is not None and self._noise_window.winfo_exists():
            self._noise_refresh_ports()
            self._noise_set_summary('Circuit changed. Run the noise analysis again.')

    def check_four_port(self):
        """Ideal branch-line benchmark for four-port networks at Inspect frequency."""
        try:
            net, _ = self._validate()
            f = float(self.inspect_ghz.get())
            report = branchline_hybrid_check(net, f*1e9)
            label = 'PASS' if report['pass'] else 'DOES NOT MEET IDEAL TARGETS'
            info = (f'Branch-line 3 dB hybrid benchmark at {f:g} GHz\n'
                    'P1 input · P2 through · P3 coupled · P4 isolated\n\n'
                    f'S11: {report["s11_db"]:.4f} dB\n'
                    f'S21: {report["s21_db"]:.4f} dB (ideal -3.0103)\n'
                    f'S31: {report["s31_db"]:.4f} dB (ideal -3.0103)\n'
                    f'S41: {report["s41_db"]:.4f} dB (isolated)\n'
                    f'S21 - S31 phase: {report["phase_difference_deg"]:+.3f}° '
                    '(ideal ±90°)\n'
                    f'Losslessness residual ||SᴴS - I||: {report["unitarity_residual"]:.3e}\n\n'
                    f'Ideal-target check: {label}\n\n'
                    'Targets: S11,S41 < -20 dB; split within 0.15 dB; '
                    'phase within 5°; unitarity < 1e-4.\n'
                    'These targets apply to a passive ideal branch-line coupler, '
                    'not arbitrary 4-port networks.')
            messagebox.showinfo('4-port branch-line verification', info, parent=self.root)
            self.status.set('Branch-line benchmark '+label+' | '+f'{f:g} GHz')
            return report
        except Exception as exc:
            messagebox.showerror('Four-port benchmark unavailable', str(exc), parent=self.root)
            return None

    def open_noise_lab(self):
        if self._noise_window is not None and self._noise_window.winfo_exists():
            self._noise_window.lift()
            self._noise_window.focus_set()
            self._noise_refresh_ports()
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        lab = tk.Toplevel(self.root)
        self._noise_window = lab
        lab.title('Topic 7 | Noise Analysis Lab v5')
        lab.geometry('1130x790')
        lab.minsize(900,650)
        lab.protocol('WM_DELETE_WINDOW', self._noise_close)
        main = ttk.Panedwindow(lab, orient='horizontal')
        main.pack(fill='both', expand=True, padx=9, pady=8)
        left = ttk.Frame(main, width=300)
        main.add(left, weight=0)
        right = ttk.Frame(main)
        main.add(right, weight=1)
        ttk.Label(left, text='Noise Analysis', font=('Helvetica',17,'bold')).pack(anchor='w',pady=(2,5))
        ttk.Label(left, text='Matched 50 Ω ports · source reference T₀ = 290 K',
                  foreground='#64748b', wraplength=285).pack(anchor='w', pady=(0,9))
        setup = ttk.LabelFrame(left, text='Analysis setup', padding=9)
        setup.pack(fill='x',pady=3)
        self.noise_src = tk.StringVar(value='P1')
        self.noise_dst = tk.StringVar(value='P2')
        self.noise_temperature = tk.StringVar(value='290')
        self.noise_start = tk.StringVar(value=self.start.get())
        self.noise_stop = tk.StringVar(value=self.stop.get())
        self.noise_points = tk.StringVar(value=min(int(self.points.get()), 201) if self.points.get().isdigit() else '101')
        self.noise_inspect = tk.StringVar(value=self.inspect_ghz.get())

        self._noise_form_row(setup, 'Source', 'src')
        self._noise_form_row(setup, 'Output', 'dst')
        self._noise_form_row(setup, 'Temperature (K)', 'temperature')
        ttk.Separator(setup).pack(fill='x', pady=8)
        for name, field in [('Start (GHz)','start'),('Stop (GHz)','stop'),('Points (≤2001)','points')]:
            self._noise_form_row(setup,name,field)
        ttk.Button(setup,text='Copy sweep from main GUI',command=self._noise_copy_sweep).pack(fill='x',pady=(8,0))

        active=ttk.LabelFrame(left,text='Active transistor model (S2P)',padding=9)
        active.pack(fill='x',pady=8)
        self.noise_use_active=tk.BooleanVar(value=False)
        self.noise_model_checkbox=ttk.Checkbutton(active,
            text='Enable USER-SUPPLIED noise assumptions',variable=self.noise_use_active)
        self.noise_model_checkbox.pack(anchor='w',pady=(0,6))
        ttk.Label(active,text='S2P has S-parameters, NOT NFmin / Rn / Γopt.\n'
                                'Defaults below are illustrative only.',
                  foreground='#9a3412',wraplength=275,justify='left',
                  font=('Helvetica',9)).pack(anchor='w',pady=(0,4))
        self.noise_nfmin=tk.StringVar(value='1.5')
        self.noise_rn=tk.StringVar(value='15')
        self.noise_gamma_mag=tk.StringVar(value='0.3')
        self.noise_gamma_phase=tk.StringVar(value='-45')
        for label, field in [('NFmin (dB)','nfmin'),('Rn (Ω)','rn'),
                             ('|Γopt|  (<1)','gamma_mag'),('∠Γopt (°)','gamma_phase')]:
            self._noise_form_row(active,label,field)

        actions=ttk.LabelFrame(left,text='Run / export',padding=9)
        actions.pack(fill='x',pady=4)
        ttk.Button(actions,text='Run noise sweep', command=self.run_noise_sweep).pack(fill='x',pady=3)
        inspected=ttk.Frame(actions); inspected.pack(fill='x',pady=3)
        ttk.Entry(inspected,textvariable=self.noise_inspect,width=8).pack(side='left')
        ttk.Label(inspected,text='GHz').pack(side='left',padx=(3,8))
        ttk.Button(inspected,text='Inspect noise',command=self.inspect_noise).pack(side='left',fill='x',expand=True)
        ttk.Button(actions,text='Export noise CSV',command=self.export_noise_csv).pack(fill='x',pady=3)

        summary=ttk.LabelFrame(right,text='Noise results and assumptions',padding=10)
        summary.pack(fill='x')
        self.noise_summary=tk.StringVar(value='Configure ports, then click Run noise sweep.\n'
                                           'Other ports are matched and thermally terminated.')
        ttk.Label(summary,textvariable=self.noise_summary,justify='left',
                  wraplength=710).pack(anchor='w')
        self.noise_figure=Figure(figsize=(8,5),dpi=100)
        self.noise_ax=self.noise_figure.add_subplot(111)
        self.noise_chart=FigureCanvasTkAgg(self.noise_figure,master=right)
        self.noise_chart.get_tk_widget().pack(side='top',expand=True,fill='both')
        self.noise_nav=NavigationToolbar2Tk(self.noise_chart,right,pack_toolbar=False)
        self.noise_nav.update();self.noise_nav.pack(side='bottom',fill='x')
        self.noise_ax.text(.5,.5,'Noise figure vs frequency',ha='center',va='center',
                           color='#64748b',transform=self.noise_ax.transAxes)
        self.noise_ax.set_xticks([]);self.noise_ax.set_yticks([])
        self.noise_chart.draw_idle()
        self._noise_refresh_ports()
        lab.lift()

    def _noise_form_row(self, parent, label, field):
        row=ttk.Frame(parent); row.pack(fill='x',pady=2)
        ttk.Label(row,text=label,width=17).pack(side='left')
        variable=getattr(self, 'noise_'+field)
        if field in ('src','dst'):
            cb=ttk.Combobox(row,textvariable=variable,state='readonly',width=11)
            cb.pack(side='right',fill='x',expand=True)
            setattr(self,'noise_'+field+'_combo',cb)
        else:
            ttk.Entry(row,textvariable=variable,width=12).pack(side='right',fill='x',expand=True)

    def _noise_set_summary(self,text):
        if hasattr(self,'noise_summary'):
            self.noise_summary.set(text)

    def _noise_copy_sweep(self):
        self.noise_start.set(self.start.get())
        self.noise_stop.set(self.stop.get())
        try:
            self.noise_points.set(str(min(int(self.points.get()),201)))
        except ValueError:
            self.noise_points.set('101')

    def _noise_refresh_ports(self):
        if self._noise_window is None or not self._noise_window.winfo_exists():
            return
        nums=sorted(self.ports)
        labels=[f'P{n}' for n in nums]
        self.noise_src_combo['values']=labels
        self.noise_dst_combo['values']=labels
        if self.noise_src.get() not in labels:
            self.noise_src.set(labels[0] if labels else '')
        if self.noise_dst.get() not in labels or self.noise_dst.get()==self.noise_src.get():
            self.noise_dst.set(labels[1] if len(labels)>1 else '')
        has_s2p=any(e['type']=='S2P' for e in self.elements)
        self.noise_model_checkbox.configure(state='normal' if has_s2p else 'disabled')
        if not has_s2p:
            self.noise_use_active.set(False)

    def _noise_close(self):
        if self._noise_window is not None:
            try:self._noise_window.destroy()
            except tk.TclError:pass
        self._noise_window=None
        self._noise_last=None
        self._noise_token+=1

    def _noise_settings(self):
        net,_=self._validate()
        try:
            src=int(self.noise_src.get().removeprefix('P'))
            dst=int(self.noise_dst.get().removeprefix('P'))
            temp=float(self.noise_temperature.get())
            params={}
            if any(e['type']=='S2P' for e in net.elements):
                if not self.noise_use_active.get():
                    raise ValueError('S2P contains no noise data. Check "Enable USER-SUPPLIED noise assumptions" first.')
                nfmin=float(self.noise_nfmin.get())
                rn=float(self.noise_rn.get())
                mag=float(self.noise_gamma_mag.get())
                phase=float(self.noise_gamma_phase.get())
                if not all(np.isfinite(x) for x in (nfmin,rn,mag,phase)):
                    raise ValueError('All S2P noise parameters must be finite numbers')
                if nfmin < 0 or rn < 0 or not (0<=mag<1):
                    raise ValueError('Require NFmin ≥ 0 dB, Rn ≥ 0 Ω, and 0 ≤ |Γopt| < 1')
                names=[e['name'] for e in net.elements if e['type']=='S2P']
                if len(names)!=1:
                    raise ValueError('One S2P device is currently supported; a separate noise model is required for each device')
                params[names[0]]=dict(nfmin_db=nfmin,rn_ohm=rn,
                    gamma_opt=mag*np.exp(1j*np.deg2rad(phase)))
            return net,src,dst,temp,params or None
        except ValueError as exc:
            raise ValueError('Invalid noise setup: '+str(exc)) from exc

    def inspect_noise(self):
        try:
            net,src,dst,temp,params=self._noise_settings()
            f=float(self.noise_inspect.get())*1e9
            point=evaluate_noise_at(net,f,src,dst,temp,params)
            report=(f'Noise inspection at {f/1e9:g} GHz · P{src} → P{dst}\n\n'
                    f'Noise figure: {point["nf_db"]:.6f} dB\n'
                    f'Noise factor: {point["f_linear"]:.7g}\n'
                    f'Transducer gain (matched): {point["gain_db"]:.5f} dB\n'
                    f'Intrinsic C_out: {point["intrinsic_cout_w_per_hz"]:.6e} W/Hz\n'
                    f'Other-port terminations: {point["termination_cout_w_per_hz"]:.6e} W/Hz\n'
                    f'Total internally generated output noise: {point["total_cout_w_per_hz"]:.6e} W/Hz\n\n'
                    f'{point["model"]}\n\n'
                    'Assumes a matched source at T₀=290 K and matched noiseless measurement load.\n'
                    'Not an absolute measured hardware noise figure.')
            self._noise_set_summary(f'At {f/1e9:g} GHz: NF {point["nf_db"]:.4f} dB, '
                   f'gain {point["gain_db"]:.4f} dB. {point["model"]}')
            messagebox.showinfo('Noise inspection',report,parent=self._noise_window)
            return point
        except Exception as exc:
            messagebox.showerror('Noise inspection error',str(exc),parent=self._noise_window)
            return None

    def run_noise_sweep(self):
        if self._noise_running:
            messagebox.showinfo('Noise sweep active','A noise sweep is already running.',parent=self._noise_window)
            return
        try:
            net,src,dst,temp,params=self._noise_settings()
            f1=float(self.noise_start.get())
            f2=float(self.noise_stop.get())
            n=int(self.noise_points.get())
            if not np.isfinite(f1) or not np.isfinite(f2) or f1<=0 or f2<f1:
                raise ValueError('Sweep requires 0 < Start ≤ Stop GHz')
            if n<1 or n>2001 or (f1==f2 and n!=1) or (n==1 and f1!=f2):
                raise ValueError('Use 1–2001 points; one-point sweep requires Start = Stop')
            freqs=np.linspace(f1*1e9,f2*1e9,n)
        except Exception as exc:
            messagebox.showerror('Invalid noise sweep',str(exc),parent=self._noise_window)
            return
        self._noise_running=True
        token=self._noise_token
        self._noise_last=None
        self._noise_set_summary(f'Computing {len(freqs)} noise samples for P{src} → P{dst} ...')
        self.status.set('Noise sweep running (solver remains unmodified) ...')

        def worker():
            try:
                values=evaluate_noise_sweep(net,freqs,src,dst,temp,params)
                self._noise_queue.put((token,'ok',values))
            except Exception as err:
                self._noise_queue.put((token,'err',str(err)))

        threading.Thread(target=worker,daemon=True).start()
        self.root.after(100,self._poll_noise)

    def _poll_noise(self):
        try:
            token,kind,payload=self._noise_queue.get_nowait()
        except queue.Empty:
            if self._noise_running:
                self.root.after(100,self._poll_noise)
            return
        self._noise_running=False
        if token!=self._noise_token or self._noise_window is None or not self._noise_window.winfo_exists():
            return
        if kind=='err':
            self._noise_set_summary('Noise sweep failed: '+payload)
            messagebox.showerror('Noise sweep failed',payload,parent=self._noise_window)
            return
        self._noise_last=list(payload)
        self._noise_render()

    def _noise_render(self):
        if not self._noise_last:return
        rows=self._noise_last
        f=np.array([r['frequency_hz']/1e9 for r in rows])
        nf=np.array([r['nf_db'] for r in rows])
        gain=np.array([r['gain_db'] for r in rows])
        self.noise_figure.clear()
        ax=self.noise_figure.add_subplot(111)
        self.noise_ax=ax
        ax2=ax.twinx()
        ax.plot(f,nf,color='#2563eb',linewidth=2,label='Noise figure (dB)',
                marker='o' if len(f)==1 else None,markersize=7)
        ax2.plot(f,gain,color='#ea580c',linewidth=1.6,linestyle='--',
                 label='Matched gain (dB)',marker='s' if len(f)==1 else None)
        if len(f)==1:
            span=max(0.01,f[0]*0.005)
            ax.set_xlim(f[0]-span,f[0]+span)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Noise figure (dB)',color='#2563eb')
        ax2.set_ylabel('Matched gain (dB)',color='#ea580c')
        ax.grid(alpha=.23)
        ax.set_title(f'Noise performance · P{rows[0]["source_port"]} → P{rows[0]["output_port"]}')
        lines1,labs1=ax.get_legend_handles_labels()
        lines2,labs2=ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2,labs1+labs2,loc='best',fontsize=9)
        self.noise_figure.tight_layout(pad=2.1)
        self.noise_chart.draw_idle()
        mid=rows[len(rows)//2]
        disclaimer=('ILLUSTRATIVE active S2P parameters (not from Touchstone).' if mid['active']
                    else 'Passive noise + matched thermal terminations for unused ports.')
        self._noise_set_summary(
            f'P{mid["source_port"]} → P{mid["output_port"]} · {len(rows)} frequencies '
            f'· {mid["temperature_k"]:g} K\n'
            f'NF range: {nf.min():.4f} to {nf.max():.4f} dB · '
            f'center {mid["frequency_hz"]/1e9:.4g} GHz: '
            f'NF {mid["nf_db"]:.4f} dB, gain {mid["gain_db"]:.4f} dB\n'
            +disclaimer)
        self.status.set(f'Noise sweep complete | {len(rows)} points | P{mid["source_port"]}→P{mid["output_port"]}')

    def export_noise_csv(self):
        if not self._noise_last:
            messagebox.showinfo('No noise results','Run the noise sweep before exporting.',parent=self._noise_window)
            return
        name=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV files','*.csv')],
                 initialfile='noise_analysis_v5.csv',parent=self._noise_window)
        if not name:return
        try:
            with open(name,'w',newline='',encoding='utf-8') as fp:
                fields=list(self._noise_last[0])
                writer=csv.DictWriter(fp,fieldnames=fields)
                writer.writeheader();writer.writerows(self._noise_last)
            self.status.set('Noise CSV exported: '+Path(name).name)
        except Exception as exc:
            messagebox.showerror('Noise CSV error',str(exc),parent=self._noise_window)


if __name__=='__main__':
    root=tk.Tk()
    app=SchematicAppV5(root)
    root.mainloop()
