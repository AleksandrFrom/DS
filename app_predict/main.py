from tkinter import *
from tkinter import messagebox

import pickle
import numpy as np

with open('autoclave_model_rf_density.pkl', 'rb') as file:
    density_model = pickle.load(file)

with open('autoclave_model_rf_thikness.pkl', 'rb') as file:
    thikness_model = pickle.load(file)

with open('autoclave_model_rf_strength.pkl', 'rb') as file:
    strength_model = pickle.load(file)

with open('autoclave_model_rf_module.pkl', 'rb') as file:
    module_model = pickle.load(file)

with open('autoclave_model_rf_lss.pkl', 'rb') as file:
    lss_model = pickle.load(file)

with open('autoclave_model_rf_Tg.pkl', 'rb') as file:
    tg_model = pickle.load(file)

def components():

    new_window = Toplevel(window)
    new_window.title('Прогнозные значения')
    new_window.geometry('500x200')
        
    l_linear_density = (float(l_linear_density_f.get()))
    l_density_yarn = (float(l_density_yarn_f.get()))
    l_strenght_yarn = (float(l_strenght_yarn_f.get()))
    l_module_yarn = (float(l_module_yarn_f.get()))
    l_lengthening = (float(l_lengthening_f.get()))
    l_mass_size = (float(l_mass_size_f.get()))
    l_loop = (float(l_loop_f.get()))
    l_density_fabric = (float(l_density_fabric_f.get()))
    l_prepreg_density = (float(l_prepreg_density_f.get()))
    l_resin_content = (float(l_resin_content_f.get()))
    l_viscosity = (float(l_viscosity_f.get()))
    l_gel_time = (float(l_gel_time_f.get()))
    l_resin_tg = (float(l_resin_tg_f.get()))
    x = np.array([[l_linear_density, l_density_yarn, l_strenght_yarn, l_module_yarn,
                  l_lengthening, l_mass_size, l_loop, l_density_fabric, l_prepreg_density,
                  l_resin_content, l_viscosity, l_gel_time,l_resin_tg]])
        
    thikness = thikness_model.predict(x)
    thikness = str(np.around(thikness, 3)[0])
    density = density_model.predict(x)
    density = str(np.around(density, 3)[0])
    strenght = strength_model.predict(x)
    strenght = str(np.around(strenght, 2)[0])
    module = module_model.predict(x)
    module = str(np.around(module, 2)[0])
    lss = lss_model.predict(x)
    lss = str(np.around(lss, 2)[0])
    tg = tg_model.predict(x)
    tg = str(np.around(tg, 2)[0])
       
    Label(new_window, text=f'Толщина монослоя: {thikness}, мм').pack(padx=2, pady=2)
    Label(new_window, text=f'Плотность: {density}, г/см3').pack(padx=2, pady=2)
    Label(new_window, text=f'Прочность при растяжении: {strenght}, МПа').pack(padx=2, pady=2)
    Label(new_window, text=f'Модуль упругости {module}, ГПа').pack(padx=2, pady=2)
    Label(new_window, text=f'Прочность при межслоевом сдвиге: {lss}, МПа').pack(padx=2, pady=2)
    Label(new_window, text=f'Температура стеклования: {tg}, 0C').pack(padx=2, pady=2)


window = Tk()
window.title('Predict')
window.geometry('800x400')


frame = Frame(window, padx=10, pady=10)
frame.pack(expand=True)

l_linear_density = Label(frame, text='Линейная плотность УВ, текс')
l_linear_density_f = Entry(frame, justify=RIGHT)
l_density_yarn = Label(frame, text='Плотность УВ, г/см3')
l_density_yarn_f = Entry(frame, justify=RIGHT)
l_strenght_yarn = Label(frame, text='Прочность при растяжении, МПа')
l_strenght_yarn_f = Entry(frame, justify=RIGHT)
l_module_yarn = Label(frame, text='Модуль упругости при растяжении, ГПа')
l_module_yarn_f = Entry(frame, justify=RIGHT)
l_lengthening = Label(frame, text='Удлинение при разрыве, %')
l_lengthening_f = Entry(frame, justify=RIGHT)
l_mass_size = Label(frame, text='Содержание аппрета, %')
l_mass_size_f = Entry(frame, justify=RIGHT)
l_loop = Label(frame, text='Прочность при разрыве в петле, сН/текс')
l_loop_f = Entry(frame, justify=RIGHT)
l_density_fabric = Label(frame, text='Поверхностная плотность ткани, г/см2')
l_density_fabric_f = Entry(frame, justify=RIGHT)
l_prepreg_density = Label(frame, text='Поверхностная плотность препрега, г/см2')
l_prepreg_density_f = Entry(frame, justify=RIGHT)
l_resin_content = Label(frame, text='Содержание связующего в препреге, %')
l_resin_content_f = Entry(frame, justify=RIGHT)
l_viscosity = Label(frame, text='Вязкость связующего, па*с')
l_viscosity_f = Entry(frame, justify=RIGHT)
l_gel_time = Label(frame, text='Время гелеобразования, мин')
l_gel_time_f = Entry(frame, justify=RIGHT)
l_resin_tg = Label(frame, text='Температура стеклования связующего')
l_resin_tg_f = Entry(frame, justify=RIGHT)
btn_submit = Button(frame, text='Ввод', command=components)

l_linear_density.grid(row=1, column=0, sticky='w', padx=2, pady=2)
l_linear_density_f.grid(row=1, column=1, sticky='e', padx=2, pady=2)
l_density_yarn.grid(row=2, column=0, sticky='w', padx=2, pady=2)
l_density_yarn_f.grid(row=2, column=1, sticky='e', padx=2, pady=2)
l_strenght_yarn.grid(row=3, column=0, sticky='w', padx=2, pady=2)
l_strenght_yarn_f.grid(row=3, column=1, sticky='e', padx=2, pady=2)
l_module_yarn.grid(row=4, column=0, sticky='w', padx=2, pady=2)
l_module_yarn_f.grid(row=4, column=1, sticky='e', padx=2, pady=2)
l_lengthening.grid(row=5, column=0, sticky='w', padx=2, pady=2)
l_lengthening_f.grid(row=5, column=1, sticky='e', padx=2, pady=2)
l_mass_size.grid(row=6, column=0, sticky='w', padx=2, pady=2)
l_mass_size_f.grid(row=6, column=1, sticky='e', padx=2, pady=2)
l_loop.grid(row=7, column=0, sticky='w', padx=2, pady=2)
l_loop_f.grid(row=7, column=1, sticky='e', padx=2, pady=2)
l_density_fabric.grid(row=8, column=0, sticky='w', padx=2, pady=2)
l_density_fabric_f.grid(row=8, column=1, sticky='e', padx=2, pady=2)
l_prepreg_density.grid(row=9, column=0, sticky='w', padx=2, pady=2)
l_prepreg_density_f.grid(row=9, column=1, sticky='e', padx=2, pady=2)
l_resin_content.grid(row=10, column=0, sticky='w', padx=2, pady=2)
l_resin_content_f.grid(row=10, column=1, sticky='e', padx=2, pady=2)
l_viscosity.grid(row=11, column=0, sticky='w', padx=2, pady=2)
l_viscosity_f.grid(row=11, column=1, sticky='e', padx=2, pady=2)
l_gel_time.grid(row=12, column=0, sticky='w', padx=2, pady=2)
l_gel_time_f.grid(row=12, column=1, sticky='e', padx=2, pady=2)
l_resin_tg.grid(row=13, column=0, sticky='w', padx=2, pady=2)
l_resin_tg_f.grid(row=13, column=1, sticky='e', padx=2, pady=2)
btn_submit.grid(row=14, column=0, columnspan=2, sticky='n', padx=5, pady=5)

window.mainloop()