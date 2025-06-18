import tkinter as tk
from tkinter import ttk

import numpy as np
import predict_helper_avt as pred

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('My App')
        self['background'] = '#EBEBEB'
        self.conf = {'padx' : (10, 30), 'pady':10}
        self.bold_font = 'Helvetica 13 bold'
        self.put_frames()

    def put_frames(self):
        self.add_form_frame = AddForm(self).grid(row=0, column=0, sticky='nswe')
        self.stat_frame = StatFrame(self).grid(row=0, column=1, sticky='nswe')
        self.table_frame = TableFrame(self).grid(row=0, column=0, 
                                                columnspan=2, sticky='nswe')

class AddForm(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self['background'] = self.master['background']
        self.items = ['autoclave, vacuum']

    def put_widgest(self):
        self.l_technology = ttk.Label(self, text='Enter technology')
        self.l_technology_f = ttk.Combobox(self, values = self.items)
        self.l_linear_density = ttk.Label(self, text='Enter linear_density')
        self.l_linear_density_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_density_yarn = ttk.Label(self, text='Enter density_yarn')
        self.l_density_yarn_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_strenght_yarn = ttk.Label(self, text='Enter strenght_yarn')
        self.l_strenght_yarn_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_module_yarn = ttk.Label(self, text='Enter module_yarn')
        self.l_module_yarn_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_lengthening = ttk.Label(self, text='Enter lengthening')
        self.l_lengthening_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_mass_size = ttk.Label(self, text='Enter mass_size')
        self.l_mass_size_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_loop = ttk.Label(self, text='Enter loop')
        self.l_loop_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_density_fabric = ttk.Label(self, text='Enter density_fabric')
        self.l_density_fabric_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_prepreg_density = ttk.Label(self, text='Enter prepreg_density')
        self.l_prepreg_density_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_resin_content = ttk.Label(self, text='Enter resin_content')
        self.l_resin_content_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_viscosity = ttk.Label(self, text='Enter viscosity')
        self.l_viscosity_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_gel_time = ttk.Label(self, text='Enter gel_time')
        self.l_gel_time_f = ttk.Entry(self, justify=tk.RIGHT)
        self.l_resin_tg = ttk.Label(self, text='Enter resin_tg')
        self.l_resin_tg_f = ttk.Entry(self, justify=tk.RIGHT)
        self.btn_submit = ttk.Button(self, text='Ввод', command=self.components)

        self.l_technology.grid(row=0, column=0, sticky='w', cnf=self.master.conf)
        self.l_technology_f.grid(row=0, column=1, sticky='e', cnf=self.master.conf)
        self.l_linear_density.grid(row=1, column=0, sticky='w', cnf=self.master.conf)
        self.l_linear_density_f.grid(row=1, column=1, sticky='e', cnf=self.master.conf)
        self.l_density_yarn.grid(row=2, column=0, sticky='w', cnf=self.master.conf)
        self.l_density_yarn_f.grid(row=2, column=1, sticky='e', cnf=self.master.conf)
        self.l_strenght_yarn.grid(row=3, column=0, sticky='w', cnf=self.master.conf)
        self.l_strenght_yarn_f.grid(row=3, column=1, sticky='e', cnf=self.master.conf)
        self.l_module_yarn.grid(row=4, column=0, sticky='w', cnf=self.master.conf)
        self.l_module_yarn_f.grid(row=4, column=1, sticky='e', cnf=self.master.conf)
        self.l_lengthening.grid(row=5, column=0, sticky='w', cnf=self.master.conf)
        self.l_lengthening_f.grid(row=5, column=1, sticky='e', cnf=self.master.conf)
        self.l_mass_size.grid(row=6, column=0, sticky='w', cnf=self.master.conf)
        self.l_mass_size_f.grid(row=6, column=1, sticky='e', cnf=self.master.conf)
        self.l_loop.grid(row=7, column=0, sticky='w', cnf=self.master.conf)
        self.l_loop_f.grid(row=7, column=1, sticky='e', cnf=self.master.conf)
        self.l_density_fabric.grid(row=8, column=0, sticky='w', cnf=self.master.conf)
        self.l_density_fabric_f.grid(row=8, column=1, sticky='e', cnf=self.master.conf)
        self.l_prepreg_density.grid(row=9, column=0, sticky='w', cnf=self.master.conf)
        self.l_prepreg_density_f.grid(row=9, column=1, sticky='e', cnf=self.master.conf)
        self.l_resin_content.grid(row=10, column=0, sticky='w', cnf=self.master.conf)
        self.l_resin_content_f.grid(row=10, column=1, sticky='e', cnf=self.master.conf)
        self.l_viscosity.grid(row=11, column=0, sticky='w', cnf=self.master.conf)
        self.l_viscosity_f.grid(row=11, column=1, sticky='e', cnf=self.master.conf)
        self.l_gel_time.grid(row=12, column=0, sticky='w', cnf=self.master.conf)
        self.l_gel_time_f.grid(row=12, column=1, sticky='e', cnf=self.master.conf)
        self.l_resin_tg.grid(row=13, column=0, sticky='w', cnf=self.master.conf)
        self.l_resin_tg_f.grid(row=13, column=1, sticky='e', cnf=self.master.conf)
        self.btn_submit.grid(row=14, column=0, columnspan=2, sticky='n', cnf=self.master.conf)
        pass 
'''
    def components():
    
    l_linear_density = (float(self.l_linear_density_f.get()))
    l_density_yarn = (float(self.l_density_yarn_f.get()))
    l_strenght_yarn = (float(self.l_strenght_yarn_f.get()))
    l_module_yarn = (float(self.l_module_yarn_f.get()))
    l_lengthening = (float(self.l_lengthening_f.get()))
    l_mass_size = (float(self.l_mass_size_f.get()))
    l_loop = (float(self.l_loop_f.get()))
    l_density_fabric = (float(self.l_density_fabric_f.get()))
    l_prepreg_density = (float(self.l_prepreg_density_f.get()))
    l_resin_content = (float(self.l_resin_content_f.get()))
    l_viscosity = (float(self.l_viscosity_f.get()))
    l_gel_time = (float(self.l_gel_time_f.get()))
    l_resin_tg = (float(self.l_resin_tg_f.get()))
    x = np.array([l_linear_density, l_density_yarn, l_strenght_yarn, l_module_yarn,
                  l_lengthening, l_mass_size, l_loop, l_density_fabric, l_prepreg_density,
                  l_resin_content, l_viscosity, l_gel_time,l_resin_tg])
    if eh.x(x):
        self.master.refresh()
    return x
'''
class StatFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self['background'] = self.master['background']

    def put_widgest(self):
        l_thiknes_monolayer_text = tk.Label(self, text='Толщина монослоя, мм', font='Arial 12 bold')
        l_thiknes_monolayer_value = tk.Label(self, text = pred.thikness_model.predict(x), font=self.master.bold_font)
        l_density_pcm_text = tk.Label(self, text='Плотность углепластика, г/см3', font='Arial 12 bold')
        l_density_pcm_value = tk.Label(self, text = pred.density_model.predict(x), font=self.master.bold_font)
        l_strenght_pcm_tetx = tk.Label(self, text='Прочность при растяжении, МПа', font='Arial 12 bold')
        l_strenght_pcm_value = tk.Label(self, text = pred.strength_model.predict(x), font=self.master.bold_font)
        l_module_pcm_text = tk.Label(self, text='Модуль упругости при растяжении, ГПа', font='Arial 12 bold')
        l_module_pcm_value = tk.Label(self, text = pred.module_model.predict(x), font=self.master.bold_font)
        l_lss_pcm_tetx = tk.Label(self, text='Прочность при межслоевом сдвиге, МПа', font='Arial 12 bold')
        l_lss_pcm_value = tk.Label(self, text = pred.lss_model.predict(x), font=self.master.bold_font)
        l_tg_pcm_tetx = tk.Label(self, text='Температура стеклования, оС', font='Arial 12 bold')
        l_tg_pcm_value = tk.Label(self, text = pred.tg_model.predict(x), font=self.master.bold_font)

        l_thiknes_monolayer_text.grid(row='0', column='0', sticky='w', cnf=self.master.conf)
        l_thiknes_monolayer_value.grid(row='0', column='1', sticky='e', cnf=self.master.conf)
        l_density_pcm_text.grid(row='1', column='0', sticky='w', cnf=self.master.conf)
        l_density_pcm_value.grid(row='1', column='1', sticky='e', cnf=self.master.conf)
        l_strenght_pcm_tetx.grid(row='2', column='0', sticky='w', cnf=self.master.conf)
        l_strenght_pcm_value.grid(row='2', column='1', sticky='e', cnf=self.master.conf)
        l_module_pcm_text.grid(row='3', column='0', sticky='w', cnf=self.master.conf)
        l_module_pcm_value.grid(row='3', column='1', sticky='e', cnf=self.master.conf)
        l_lss_pcm_tetx.grid(row='4', column='0', sticky='w', cnf=self.master.conf)
        l_lss_pcm_value.grid(row='4', column='1', sticky='e', cnf=self.master.conf)
        l_tg_pcm_tetx.grid(row='5', column='0', sticky='w', cnf=self.master.conf)
        l_tg_pcm_value.grid(row='5', column='1', sticky='e', cnf=self.master.conf)
        pass 

class TableFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self['background'] = self.master['background']

    def put_widgest(self):
        pass 



app = App()
app.mainloop()