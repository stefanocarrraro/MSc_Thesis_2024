#%% Complete model version 3 - 22/04/2024

import pyomo.environ as pyo 
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import os
import time
import pickle
from datetime import datetime, timedelta

def read_input_file(filename, filename_to_save, re_run=False):
    if re_run:
        weather = pd.read_excel(filename, sheet_name='weather', index_col=0)
        bess = pd.read_excel(filename, sheet_name= 'BESS')
        pv_data = pd.read_excel(filename, sheet_name='PV', index_col=0)
        ev_data = pd.read_excel(filename, sheet_name= 'EV_data', index_col=0)
        heating = pd.read_excel(filename, sheet_name= 'heating_system', index_col=0)
        transformers = pd.read_excel(filename, sheet_name='transformers', index_col=0)
        load = pd.read_excel(filename, sheet_name= 'load', index_col=0)

        house = {}
        for name in load.index: #load_type:
            house[name] = pd.read_excel(filename, sheet_name=name, index_col=0)


        res_dict =  {'ev_data':ev_data, 'pv_data':pv_data, 'heating':heating, 'load':load, \
                'bess':bess, 'weather':weather, 'transformers':transformers, 'house':house}
        
        pickle.dump(res_dict, open(filename_to_save, 'wb'))
        return res_dict
    else:
        return pickle.load(open(filename_to_save, 'rb'))

def create_model():
    # Building the base structure of the model
    # NOTE: no objective function and constraints defined for now

    # Create the model:
    m = pyo.AbstractModel()

    # Error check:
    m.error = pyo.Param(initialize=False, domain=pyo.Binary, mutable=True)              # Turns positive when an error occurs


    # ----------------------------
    # SETS:
    # ----------------------------

    m.T = pyo.Set(ordered=True, doc='EVs')                                              # [YYYY/MM/DD HH:MM, ...]                                       
    m.S = pyo.Set(ordered=True, doc='EVs')                                              # [EV1, EV2, ..., EVs]
    m.L = pyo.Set(ordered=True, doc='Loads')                                            # [House type 1, House type l]
    m.P = pyo.Set(ordered=True, doc='PVs')                                              # [PV1, PV2, ..., PVp]
    m.H = pyo.Set(ordered=True, doc='Heat pump goups')                                  # [HP1, HP2, ..., HPh]
    m.Tr = pyo.Set(ordered=True, doc='Transformers')                                    # [TRAFO1, ..., TRAFOtr]

    # ----------------------------
    # PARAMETERS:
    # ----------------------------

    # Set values for trade-offs
    m.SET_EV = pyo.Param(domain=pyo.NonNegativeReals, mutable=True)                     # EVs case

    # BESS:
    m.BESS_capacity = pyo.Param(domain=pyo.NonNegativeReals, mutable=True)              # Charging volume of b-th battery [kWh]
    m.BESS_Emin = pyo.Param(domain=pyo.NonNegativeReals, mutable=True)                  # Min SOC [%]
    m.BESS_Emax = pyo.Param(domain=pyo.NonNegativeReals, mutable=True)                  # Max SOC [%]
    m.BESS_Pmax = pyo.Param(domain=pyo.NonNegativeReals, mutable=True)                  # Max power of charge [kW]
    m.BESS_Pmin = pyo.Param(domain=pyo.NonPositiveReals, mutable=True)                  # Min power of charge [kW]
   
    # Constants:
    m.Cp_w = pyo.Param(initialize=4.186, domain=pyo.NonNegativeReals, mutable=True)     # Specific heat of water [kJ/(kg K)]
    m.m_flow = pyo.Param(initialize=150, domain=pyo.NonNegativeReals, mutable=True)     # Mass flow rate in the heating system [kg/s]

    # Heating system + PCM Thermal storage
    m.P_HP_rated = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)            # Power rated of heatpumps h [kW]
    m.T_m = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)                   # melting temperature of the PCM [°C]
    m.c_pcm = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)                 # Specific heat of the PCM used [kJ/kgK]
    m.m_pcm = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)                 # Mass of the PCM [kg]
    m.h_pcm = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)                 # Latent heat [kJ/kg]
    m.E_s = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)                   # Upper bound of the solid phase [kWh]
    m.E_l = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)                   # Lower bound of the liquid phase [kWh]
    m.E_pcm_min = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)             # Min capacity of the pcm [kWh]
    m.E_pcm_max = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)             # Max capacity of the pcm [kWh]
    m.T_max = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)                 # maximum temperature allowed in the PCM [°C]
    m.T_min = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)                 # minimum temperature allowed in the PCM [°C]
    m.COP = pyo.Param(m.T, domain=pyo.NonNegativeReals, mutable=True)                   # COP of heat pumps at time t [-]
    m.delta_T = pyo.Param(m.H, domain=pyo.NonNegativeReals, mutable=True)               # Maximum delta T in time for the heating system [°C/dt]


    # EV:
    m.EV_Emax = pyo.Param(m.S, domain=pyo.NonNegativeReals, mutable=True)               # Charging volume of vehicle s [kWh]
    m.EV_Pmax = pyo.Param(m.S, domain=pyo.NonNegativeReals, mutable=True)               # Max charging power of veichle s [kW]
    m.EV_Pmin = pyo.Param(m.S, domain=pyo.Reals, mutable=True)                          # Min charging power of vehicle s[kW]
    m.Tarr = pyo.Param(m.S, domain=pyo.Any, mutable=True)                               # Arrival time of veichle s [h]
    m.Tdep = pyo.Param(m.S, domain=pyo.Any, mutable=True)                               # Departure time of vehicle s [h]

    # Loads:
    m.N_houses = pyo.Param(m.L, domain=pyo.NonNegativeIntegers, mutable=True)           # Number of houses for house type k [-]
    m.E_load = pyo.Param(m.T, m.L, domain=pyo.NonNegativeReals, mutable=True)           # Electrical load of house k at time t [kW]
    m.Q_hw = pyo.Param(m.T, m.L, domain=pyo.NonNegativeReals, mutable=True)             # Hot water demand for house l at time t [kW]
    m.Q_sh = pyo.Param(m.T, m.L, domain=pyo.NonNegativeReals, mutable=True)             # Space heating demand for house l at time t [kW]   
    m.Q_cool = pyo.Param(m.T, m.L, domain=pyo.NonPositiveReals, mutable=True)           # Space cooling demand for house l ad time t [kW]
    m.T_max_sh = pyo.Param(domain= pyo.NonNegativeReals, mutable=True)                  # Max temperature for space heating in house l [°C]
    m.T_min_sh = pyo.Param(domain= pyo.NonNegativeReals, mutable=True)                  # Min temperature for space heating in house l [°C]
    m.T_max_hw = pyo.Param(domain= pyo.NonNegativeReals, mutable=True)                  # Max temperature for hot water in house l [°C]
    m.T_min_hw = pyo.Param(domain= pyo.NonNegativeReals, mutable=True)                  # Min temperature for hot water in house l [°C]
    m.T_max_cool = pyo.Param(domain= pyo.NonNegativeReals, mutable=True)                # Max temperature for cooling in house l [°C]
    m.T_min_cool = pyo.Param(domain= pyo.NonNegativeReals, mutable=True)                # Min temperature for cooling in house l [°C]

    # Peak hours:
    m.peak_schedule = pyo.Param(m.T, domain=pyo.Binary, mutable=True)                   # t is (is not) peak hour
    m.P_PEAK = pyo.Param(domain=pyo.NonNegativeReals, mutable=True)                     # max allowable power during peak hours [kW]

    # PV:
    m.PV_peak = pyo.Param(m.P, domain=pyo.NonNegativeReals, mutable=True)               # p-th string's module peak power [kWp]
    # m.PV_az = pyo.Param(m.P, domain=pyo.Integers, mutable=True)                         # p-th pv string's azimuth angle [rad]
    # m.PV_theta = pyo.Param(m.P, domain=pyo.NonNegativeReals, mutable=True)              # p-th pv string's tilt angle [rad]
    m.PV_N_M = pyo.Param(m.P, domain=pyo.NonNegativeIntegers, mutable=True)             # Number of modules in string p [-]
    # m.PV_A_M = pyo.Param(m.P, domain=pyo.NonNegativeReals, mutable=True)                # Area of modules in string p [m2]
    # m.PV_albedo = pyo.Param(m.P, domain=pyo.Reals, mutable=True)                        # p-th pv string albedo [-]
    # m.PV_eff = pyo.Param(m.P, domain=pyo.Reals, mutable=True)                           # p-th pv string efficiency [-]

    # Weather:
    # m.GHI = pyo.Param(m.T, domain=pyo.NonNegativeReals, mutable=True)                   # Global horizontal irradiation at time t [kW/m2]
    # m.DNI = pyo.Param(m.T, domain=pyo.NonNegativeReals, mutable=True)                   # Direct Normal irradiance at time t [kW/m2]
    # m.DIF = pyo.Param(m.T, domain=pyo.NonNegativeReals, mutable=True)                   # Diffuse horizontal irradiance at time t [kW/m2]
    m.T_outdoor = pyo.Param(m.T, domain=pyo.Reals, mutable=True)                        # Outdoor T at time t [°C]
    # m.Sun_elev = pyo.Param(m.T, domain=pyo.Reals, mutable=True)                         # Sun elevation at time t [m]
    m.PV_output = pyo.Param(m.T, domain=pyo.NonNegativeReals, mutable=True)             # Power output of a 1kWp pv system at time t [W/1kWp]

    # Transformer:
    m.trafo_limit = pyo.Param(m.Tr, domain=pyo.NonNegativeIntegers, mutable=True)       # Physical limit for power withdraw
    
    

    # ----------------------------
    # VARIABLES:
    # ----------------------------

    # BESS:
    m.e_BESS = pyo.Var(m.T, domain=pyo.NonNegativeReals)        # [kWh]                 # State of Energy of battery b at time t
    m.p_BESS_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)     # [kW]                  # power of discharge(charge) of battery b at time t
    m.p_BESS_dch = pyo.Var(m.T, domain=pyo.NonPositiveReals)    # [kW]                  # power of discharge(charge) of battery b at time t
    m.u_BESS = pyo.Var(m.T, domain=pyo.Binary)                  # 0/1                   # Binary variable for the battery is discharing

    # EVs:
    m.p_EV = pyo.Var(m.T, m.S, domain=pyo.Reals)                # [kW]                  # Charging/discharging power to vehicle s at time t
    m.e_EV = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)     # [kWh]                 # State of Energy of vehicle s at time t
    m.ev_soft = pyo.Var(m.S, domain=pyo.Reals, bounds=(0,1))    # [-]                   # softener of EV final charge constraint

    # Loads:
    m.p_el = pyo.Var(m.T, m.L, domain=pyo.NonNegativeReals)     # [kW]                  # Power demand of the l-th load at time t  
    
    # Peak hours:
    m.p_tot = pyo.Var(m.T, domain=pyo.Reals)                    # [kW]                  # Total power at each time t
    m.p_peak = pyo.Var(domain=pyo.NonNegativeReals)             # [kW]                  # Max power during peak hours
    
    # PV:
    m.p_PV = pyo.Var(m.T, m.P, domain=pyo.NonNegativeReals)     # [kW]                  # Power output from the p-th PV string 
    
    # Heating system + PCM thermal storage:
    m.p_HP = pyo.Var(m.T, m.H, domain=pyo.NonNegativeReals)     # [kW]                  # Power needed by the h-th HP at time t 
    m.q_HP = pyo.Var(m.T, m.H, domain=pyo.Reals)                # [kW]                  # Heating power of each heatpump at time t

    m.T1 = pyo.Var(m.T, m.H, domain=pyo.Reals)                  # [°C]                  # Temperature after HP
    m.T2 = pyo.Var(m.T, m.H, domain=pyo.Reals)                  # [°C]                  # Temperature before load
    m.T3 = pyo.Var(m.T, m.H, domain=pyo.Reals)                  # [°C]                  # Temperature after load
    m.T4 = pyo.Var(m.T, m.H, domain=pyo.Reals)                  # [°C]                  # Temperature before HP
    m.T_pcm = pyo.Var(m.T, m.H ,domain=pyo.Reals)               # [°C]                  # Temperature of PCM

    m.u_s = pyo.Var(m.T, m.H, domain=pyo.Binary)                # 0/1                   # Binary variable solid phase PCM active
    m.u_l = pyo.Var(m.T, m.H, domain=pyo.Binary)                # 0/1                   # Binary variable liquid phase PCM active
    m.u_12 = pyo.Var(m.T, m.H, domain=pyo.Binary)               # 0/1                   # Binary variable for the condition T1 >= T_PCM
    m.u_34 = pyo.Var(m.T, m.H, domain=pyo.Binary)               # 0/1                   # Binary variable for the condition T3 >= T_PCM

    m.q_12 = pyo.Var(m.T, m.H, domain=pyo.Reals)                # [kW]                  # positive entering in the water (exiting the hot side) [kW]
    m.q_34 = pyo.Var(m.T, m.H, domain=pyo.Reals)                # [kW]                  # positive entering in the water (exiting the cold side) [kW]
    m.e_pcm = pyo.Var(m.T, m.H, domain=pyo.NonNegativeReals)    # [kWh]                 # SOE respect to the minimum ammissible [kWh]

    # ----------------------------
    # OBJECTIVE:
    # ----------------------------
    m.obj = pyo.Objective()

    return m

def initialize_model(abs_model, data, day, delta):
    # Initialize the part of the model constant for all cases
    m = abs_model.create_instance()

    def update_sets(m, data, day, delta):                       # updating all sets for selected day

        def time_set(m, start_date, delta):
            if delta == 'hours' or delta == 'Hours':
                m.T = range(72)
                return [start_date + timedelta(hours=i) for i in range(72)]
            elif delta == '30 min':
                m.T = range(72*2)
                return [start_date + timedelta(minutes=i) for i in range(0, 72*60, 30)]
            elif delta == '15 min':
                m.T = range(72*4)
                return [start_date + timedelta(minutes=i) for i in range(0, 72*60, 15)]
            else:
                print('\n-------------------------\nERROR: wrong m.T definition\n-------------------------\n')

        def EV_set(data, day):
            ev_data = data['ev_data']
            start_date =  day - timedelta(hours=24)  # 00:00 of the day before "day"
            end_date = day + timedelta(hours=24)    # 00:00 of the day after "day"
            set_S = []

            for s in data['ev_data'].index:
                Tarr = pd.Timestamp.to_pydatetime(ev_data['T arrival [h]'][s])
                # Tdep = ev_data['T departure [h]'][s]

                if Tarr > start_date and Tarr < end_date:
                    set_S.append(s)
            return set_S

        start_date = day - timedelta(days=1)                    # 00:00 of the day before "day"

        time_dt = time_set(m, start_date, delta)                # [0h, 1h, ..., Th]
        m.S = EV_set(data, day)                                 # [EV1, EV2, ..., EVs]                
        m.L = data['load'].index                                # ['House type 1', ..., 'House type l']
        m.P = data['pv_data'].index                             # [PV1, PV2, ..., PVp]
        m.H = data['heating'].index                             # [HP1, HP2, ..., HPh]
        m.Tr = data['transformers'].index                       # [TRAFO1, ..., TRAFOtr]
        if len(time_dt) != len(m.T):
            m.error = True
            print('------ ERROR: wrong time set construction ------')
        else:
            time = [x.strftime('%m/%d/%Y %H:%M:%S') for x in time_dt]
            return time
    time = update_sets(m, data, day, delta)
    
    def update_data(m, data, time, day):                        # impoerting all data for selected day
        
        def update_weather(m, data, time):
            weather = data['weather']
            for t in m.T:
                if time[t] in weather.index:
                    # m.GHI[t] = weather['GHI [W/m2]'][time[t]]/1000                          # [kW/m2]
                    # m.DNI[t] = weather['DNI [W/m2]'][time[t]]/1000                          # [kW/m2]
                    # m.DIF[t] = weather['DIF [W/m2]'][time[t]]/1000                          # [kW/m2]
                    m.T_outdoor[t] = weather['T outdoor [°C]'][time[t]]                     # [°C]
                    # m.Sun_elev[t] = weather['H sun [m]'][time[t]]                           # [m]
                    m.PV_output[t] = weather['PV output [W/1kWp]'][time[t]]                 # [W/1kWp]
                    m.COP[t] = weather['COP'][time[t]]                                      # [-]

                else:
                    m.error = True
                    print(f'------ ERROR: weather data time {time[t]} missing ------')
                    break              

            # Unfeasibility check
            def unfeasibility_check(m):
                def check(set, to_check, m=m):
                    for s in set:
                        if to_check[s].value is None:
                            m.error = True
                            break  
                # check(m.T, m.GHI)          
                # check(m.T, m.DNI)
                # check(m.T, m.DIF)
                check(m.T, m.T_outdoor)
                # check(m.T, m.Sun_elev)
                check(m.T, m.PV_output)
                check(m.T, m.COP)
                if m.error.value:
                    print('------ ERROR: weather data missing ------')
            unfeasibility_check(m)
        update_weather(m, data, time)

        def update_ev(m, data, day, delta):
            ev_data = data['ev_data']

            for s in m.S:
                m.EV_Pmax[s] = ev_data['Max charging power [kW]'][s]            # Max charging power [kW]
                m.EV_Pmin[s] = ev_data['Min charging power [kW]'][s]            # Min charging power [kW]

                Tarr = ev_data['T arrival [h]'][s]
                Tdep = ev_data['T departure [h]'][s]

                m.Tarr[s] = Tarr

                if (Tdep - m.Tarr[s].value).total_seconds() / 3600 > 24:
                    m.Tdep[s] = m.Tarr[s].value + timedelta(hours=24)     # The vehicle can charge max for 24h
                else:
                    m.Tdep[s] = Tdep

                if (((m.Tdep[s].value - m.Tarr[s].value).total_seconds() / 3600)* \
                    m.EV_Pmax[s].value < ev_data['Charging volume [kWh]'][s]):

                    deltaT = (m.Tdep[s].value - m.Tarr[s].value)
                    deltaT = deltaT.total_seconds() / 3600
                    m.EV_Emax[s] = deltaT*m.EV_Pmax[s].value
                else:
                    m.EV_Emax[s] = ev_data['Charging volume [kWh]'][s]              # Charging volume [kWh]

            # Unfesibility check 
            def unfeasibiility_check(m):
                def check(set, to_check, m=m):
                    for s in set:
                        if to_check[s].value is None:
                            m.error = True
                            break            
                check(m.S, m.Tarr)
                check(m.S, m.Tdep)
                check(m.S, m.EV_Pmax)
                check(m.S, m.EV_Emax)
                check(m.S, m.EV_Pmin)
                if m.error.value:
                    print('------ ERROR: EV data missing ------')
            unfeasibiility_check(m)
        update_ev(m, data, day, delta)

        def update_trafo(m, data):
            trafo = data['transformers']
            for tr in m.Tr:
                m.trafo_limit[tr] = trafo['Power limit [kW]'][tr]

            # Unfeasibility check
            def unfeasibility_check(m):
                def check(set, to_check, m=m):
                    for s in set:
                        if to_check[s].value is None:
                            m.error = True
                            break            
                check(m.Tr, m.trafo_limit)
                if m.error.value:
                    print('------ ERROR: transformer data missing ------')
            unfeasibility_check(m)
        update_trafo(m, data)
               
        def update_heating(m, data):
            heating = data['heating']
            for h in m.H:
                m.P_HP_rated[h] = heating['Power rated [kW]'][h]      # [kW]
                m.T_max[h] = heating['T max [°C]'][h]                 # [°C]
                m.T_min[h] = heating['T min [°C]'][h]                 # [°C]
                m.c_pcm[h] = heating['PCM capacity [kJ/kgK]'][h]      # [kJ/kgK]
                m.m_pcm[h] = heating['PCM mass [kg]'][h]              # [kg]
                m.h_pcm[h] = heating['PCM latent heat [kJ/kg]'][h]    # [kJ/kg]
                m.T_m[h] = heating['T melting [°C]'][h]               # [°C]
                m.delta_T[h] = heating['Max temperature change [°C/dt]'][h]

                # To set e_pcm relative to Celsius degrees:

                # kWh = kg * K * kJ/kgK * h/s 
                E_pcm_0 = m.m_pcm[h].value * (273.15) * m.c_pcm[h].value / 3600               # [kWh]

                # kWh = kg * K * kJ/kgK * h/s
                E_pcm_min = m.m_pcm[h].value * (273.15 + m.T_min[h].value) * m.c_pcm[h].value / 3600               # [kWh]
                m.E_pcm_min[h] = E_pcm_min - E_pcm_0

                # kWh = kg * K * kJ/kgK * h/s
                E_s = m.m_pcm[h].value * (273.15 + m.T_m[h].value) * m.c_pcm[h].value / 3600                       # [kWh]
                m.E_s[h] = E_s - E_pcm_0

                # kWh = kWh + kg * kJ/kg * h/s
                E_l = E_s + m.m_pcm[h].value * m.h_pcm[h].value / 3600                                  # [kWh]
                m.E_l[h] = E_l - E_pcm_0

                # kWh = kg * (K * kJ/kgK + kJ/kg) * h/s
                E_pcm_max = m.m_pcm[h].value * ((273.15 + m.T_max[h].value) * m.c_pcm[h].value + m.h_pcm[h].value) / 3600     # [kWh]
                m.E_pcm_max[h] = E_pcm_max - E_pcm_0

                # print('E_pcm', m.E_pcm_min[h].value, m.E_s[h].value, m.E_l[h].value, m.E_pcm_max[h].value)

            # Unfesibility check 
            def unfeasibility_check(m):
                def check(set, to_check, set2=0, m=m):       
                    if set2 == 0:
                        for s in set:
                            if to_check[s].value is None:
                                m.error = True
                                break            
                    elif set2 != 0:
                        for s in set:
                            for r in set2:
                                if to_check[s,r].value is None:
                                    m.error = True
                                    break     
                check(m.H, m.P_HP_rated)
                check(m.H, m.delta_T)
                check(m.H, m.c_pcm)
                check(m.H, m.m_pcm)
                check(m.H, m.h_pcm)
                check(m.H, m.T_m)
                check(m.H, m.E_s)
                check(m.H, m.E_l)
                check(m.H, m.E_pcm_min)
                check(m.H, m.E_pcm_max)
                if m.error.value:
                    print('------ ERROR: Heatpumps data missing ------')
            unfeasibility_check(m)
        update_heating(m, data)        

        def update_demand(m, data, time):
            house = data['house']
            load = data['load']

            m.T_max_sh = max(load['T max space heating [°C]'][l] for l in m.L) 
            m.T_min_sh = max(load['T min space heating [°C]'][l] for l in m.L) 
            m.T_max_hw = max(load['T max hot water [°C]'][l] for l in m.L)  
            m.T_min_hw = max(load['T min hot water [°C]'][l] for l in m.L) 
            m.T_max_cool = max(load['T max cooling [°C]'][l] for l in m.L) 
            m.T_min_cool = max(load['T min cooling [°C]'][l] for l in m.L)

            for l in m.L:
                    house_type = house[l]
                    m.N_houses[l] = load['Number'][l]
                    for t in m.T:
                        m.Q_hw[t,l] = house_type['Hot water load [W/house]'][time[t]]/1000 * m.N_houses[l].value            # [kW]
                        m.Q_sh[t,l] = house_type['Space heating load [W/house]'][time[t]]/1000 * m.N_houses[l].value        # [kW]
                        m.E_load[t,l] = house_type['Electrical load [W/house]'][time[t]]/1000 * m.N_houses[l].value         # [kW]
                        m.Q_cool[t,l] = - house_type['Space cooling load [W/house]'][time[t]]/1000 * m.N_houses[l].value    # [kW]

                        m.peak_schedule[t] = house_type['Peak hours'][time[t]]          
            

            # Unfesibility check (checks if all parameters have a value)
            def unfeasibility_check(m):
                def check(set, to_check, set2=0, m=m):
                    if set2 == 0:
                        for s in set:
                            if to_check[s].value is None:
                                m.error = True
                                break            
                    elif set2 != 0:
                        for s in set:
                            for r in set2:
                                if to_check[s,r].value is None:
                                    print(to_check[s,r].value)
                                    m.error = True
                                    break      
                check(m.T, m.Q_hw, m.L)
                check(m.T, m.Q_sh, m.L)
                check(m.T, m.peak_schedule)
                check(m.T, m.E_load, m.L)
                check(m.T, m.Q_cool, m.L)
                if m.error.value:
                    print('------ ERROR: missing data in loads ------')
            unfeasibility_check(m)
        update_demand(m, data, time)

        def update_pv(m, data):
            pv = data['pv_data']
            for p in m.P:
                # m.PV_az[p] = pv['Azimuth (south 0 degrees, clockwise)'][p]  * (np.pi / 180)     # [rad]
                # m.PV_theta[p] = pv['Tilt [degrees]'][p] * (np.pi / 180)                         # [rad]
                # m.PV_A_M[p] = pv['Module area [m2]'][p]
                m.PV_N_M[p] = pv['Number of panels'][p]
                # m.PV_albedo[p] = pv['Albedo'][p]
                m.PV_peak[p] = m.PV_N_M[p] * pv['Peak power [kWp/panel]'][p]/1000               # [kWp]

                # # Defining the panel efficiency:
                # G_stc = 1       # [kW/m2]
                # m.PV_eff[p] = pv['Peak power [kWp/panel]'][p] / (G_stc * m.PV_A_M[p].value)     # [-]

            # Unfeasibility check
            def unfeasibility_check(m):
                def check(set, to_check, set2=0, m=m):       
                    if set2 == 0:
                        for s in set:
                            if to_check[s].value is None:
                                m.error = True
                                break            
                    elif set2 != 0:
                        for s in set:
                            for r in set2:
                                if to_check[s,r].value is None:
                                    m.error = True
                                    break     
                check(m.P, m.PV_peak)
                # check(m.P, m.PV_az)
                # check(m.P, m.PV_theta)
                # check(m.P, m.PV_A_M)
                check(m.P, m.PV_N_M)
                # check(m.P, m.PV_albedo)
                # check(m.P, m.PV_eff)
                if m.error.value:
                    print('------ ERROR: PV data missing ------')
            unfeasibility_check(m)
        update_pv(m, data)

        def update_bess(m, data):
            bess = data['bess']

            m.BESS_capacity = bess['Battery capacity [kWh]'][0]
            m.BESS_Emax = bess['Max SOC [%]'][0] * m.BESS_capacity.value / 100      # SOE
            m.BESS_Emin = bess['Min SOC [%]'][0] * m.BESS_capacity.value / 100      # SOE
            m.BESS_Pmax = bess['Max charging power [kW]'][0]
            m.BESS_Pmin = - bess['Max discharging power [kW]'][0]

            # Unfeasibility check
            def unfeasibility_check(m):
                def check(to_check, m=m):
                    if to_check.value is None:
                        m.error = True        
                check(m.BESS_capacity)
                check(m.BESS_Emax)
                check(m.BESS_Emin)
                check(m.BESS_Pmax)
                check(m.BESS_Pmin)
                if m.error.value:
                    print('------ ERROR: BESS data missing ------')
            unfeasibility_check(m)
        update_bess(m, data)
    update_data(m, data, time, day)

    def update_constraints(m, time, day, delta):                # setting all constraints for the selected day
        # ----------------------------
        # CONSTRAINTS: 
        # ----------------------------

        # PV:
        # def PV_output_precise(m, t, p):                                     # PV power output
        #     G_dir_t_p = m.DNI[t].value * (np.sin(m.PV_theta[p].value) * np.cos(m.Sun_elev[t].value) * \
        #                 np.cos(m.PV_az[p].value - m.Sun_az[t].value) + 
        #                 np.cos(m.PV_theta[p].value) * np.sin(m.Sun_elev[t].value))
            
        #     svf_p = (1 + np.cos(m.PV_theta[p].value))/2
        #     G_dif_t_p = m.DIF[t].value * svf_p

        #     ghi_t_p = m.DNI[t].value * np.cos(m.Sun_elev[t].value) + m.DIF[t].value
        #     G_ground_t_p = ghi_t_p * m.PV_albedo[p].value * (1 - svf_p)

        #     G_tot_t_p = G_dir_t_p + G_dif_t_p + G_ground_t_p

        #     return m.p_PV[t,p] == m.PV_eff[p] * G_tot_t_p * m.PV_A_M[p] * m.PV_N_M[p]
        # m.PV_power_balance = pyo.Constraint(m.T, m.P, rule=PV_output_precise)

        def PV_output(m, t, p):                                             # PV power output
            return m.p_PV[t,p] == m.PV_output[t] * m.PV_peak[p]
        m.PV_output_con = pyo.Constraint(m.T, m.P, rule=PV_output)


        # Loads:
        def E_load_balance(m, t, l):                                        # E_load power balance constraint at time t for house l [kW]
            return m.p_el[t,l] == m.E_load[t,l]
        m.E_load_balance_con = pyo.Constraint(m.T, m.L, rule=E_load_balance)

        def LT_demand_power(m, t):                                          # H_load power balance in LT network
            if sum(m.Q_cool[t,l].value for l in m.L) <= 0 and (m.T_outdoor[t].value <= 15):       # No cooling demand -> heating mode
                # Space heating
                Q_demand = sum(m.Q_sh[t,l].value for l in m.L)
            else:
                # Cooling
                Q_demand = sum(m.Q_cool[t,l].value for l in m.L) # negative since heat is absorbed
            
            return Q_demand == m.m_flow * m.Cp_w * (m.T2[t,'LT'] - m.T3[t,'LT'])
        m.LT_demand_power_con = pyo.Constraint(m.T, rule=LT_demand_power)

        def HT_demand_power(m, t):                                          # H_load power balance in HT network
            if sum(m.Q_cool[t,l].value for l in m.L) <= 0 and (m.T_outdoor[t].value <= 15):       # No cooling demand -> heating mode
                # hot water
                Q_demand = sum(m.Q_hw[t,l].value for l in m.L)
            else:                                                                       # Cooling mode
                # Space heating and hot water
                Q_demand = sum((m.Q_sh[t,l].value + m.Q_hw[t,l].value) for l in m.L)
            
            return Q_demand == m.m_flow * m.Cp_w * (m.T2[t,'HT'] - m.T3[t,'HT'])
        m.HT_demand_power_con = pyo.Constraint(m.T, rule=HT_demand_power)

        def LT_heating_balance(m, t):                                       # Power balance in LT network
            if sum(m.Q_cool[t,l].value for l in m.L) <= 0 and (m.T_outdoor[t].value <= 15):      # No cooling demand -> heating mode
                Q_demand = sum(m.Q_sh[t,l].value for l in m.L)
            else:                                                                   # Cooling mode
                Q_demand = sum(m.Q_cool[t,l].value for l in m.L) 
            return Q_demand == m.q_HP[t,'LT'] + (m.q_12[t,'LT'] + m.q_34[t,'LT'])
        m.LT_heating_balance_con = pyo.Constraint(m.T, rule=LT_heating_balance)

        def HT_heating_balance(m, t):                                       # Power balance in HT network
            if sum(m.Q_cool[t,l].value for l in m.L) <= 0 and (m.T_outdoor[t].value <= 15):     # No cooling demand -> heating mode
                Q_demand = sum(m.Q_hw[t,l].value for l in m.L)
            else:                                                                               # Cooling mode
                Q_demand = sum((m.Q_sh[t,l].value + m.Q_hw[t,l].value) for l in m.L)
            return Q_demand == m.q_HP[t,'HT'] + (m.q_12[t,'HT'] + m.q_34[t,'HT'])
        m.HT_heating_balance_con = pyo.Constraint(m.T, rule=HT_heating_balance)


        # Heating system + PCM thermal storage
        def HP_power(m, t, h):                                              # Power delivered my HPs
            return m.q_HP[t,h] == m.m_flow * m.Cp_w * (m.T1[t,h] - m.T4[t,h])
        m.HP_power_con = pyo.Constraint(m.T, m.H, rule=HP_power)

        def pcm_1(m, t, h):                                                 # phase <= constraint
            k = 100000
            return  m.e_pcm[t,h] <= m.E_l[h] + (m.E_s[h] - m.E_l[h]) * m.u_s[t,h] + k * m.u_l[t,h]
        m.pcm_con1 = pyo.Constraint(m.T, m.H, rule=pcm_1)

        def pcm_2(m, t, h):                                                 # phase >= constraint
            k = 100000
            return  m.e_pcm[t,h] >= m.E_s[h] + (m.E_l[h] - m.E_s[h]) * m.u_l[t,h] - k * m.u_s[t,h]
        m.pcm_con2 = pyo.Constraint(m.T, m.H, rule=pcm_2)

        def pcm_3(m, t, h):                                                 # u_s + u_l <= 1
            return m.u_l[t,h] + m.u_s[t,h] <= 1
        m.pcm_con3 = pyo.Constraint(m.T, m.H, rule=pcm_3)

        def pcm_4(m, t, h):                                                 # T_pcm depending on e_pcm and phase
            return m.T_pcm[t,h] == 1 / (m.c_pcm[h]) * (m.e_pcm[t,h]*3600/(m.m_pcm[h]) - m.h_pcm[h] * m.u_l[t,h])  * \
                    (m.u_l[t,h] + m.u_s[t,h]) + m.T_m[h] * (1 - m.u_l[t,h] - m.u_s[t,h])
        m.pcm_con4 = pyo.Constraint(m.T, m.H, rule=pcm_4)

        def pcm_5(m, t, h):                                                 # chargin/discharging of pcm and INITIAL SOE 
            if t > 0:
                return m.e_pcm[t,h] ==0.95* m.e_pcm[t-1,h] - (m.q_12[t,h] + m.q_34[t,h]) * 1
            else:
                # initial condition: empty PCM
                return m.e_pcm[t,h] == m.e_pcm[71,h]
        m.pcm_con5 = pyo.Constraint(m.T, m.H, rule=pcm_5)

        def hot_side_1(m, t, h):                                            # Constraint for condition T1 >= T_pcm
                w = 100000
                return m.T1[t,h] <= m.T_pcm[t,h] + m.u_12[t,h] * w
        m.hot_side_con1 = pyo.Constraint(m.T, m.H, rule=hot_side_1)

        def hot_side_2(m, t, h):                                            # Constraint for condition T1 <= T_pcm
                w = 100000
                return m.T1[t,h] >= m.T_pcm[t,h] - (1 - m.u_12[t,h]) * w
        m.hot_side_con2 = pyo.Constraint(m.T, m.H, rule=hot_side_2)

        def hot_side_3(m, t, h):                                            # T2 in between T1 and T_pcm
            return m.T2[t,h] >= m.T_pcm[t,h] * m.u_12[t,h] + (1 - m.u_12[t,h]) * m.T1[t,h]
        m.hot_side_con3 = pyo.Constraint(m.T, m.H, rule=hot_side_3)

        def hot_side_4(m, t, h):                                            # T2 in between T_pcm and T1                                        
            return m.T2[t,h] <= m.T1[t,h] * m.u_12[t,h] + (1 - m.u_12[t,h]) * m.T_pcm[t,h]
        m.hot_side_con4 = pyo.Constraint(m.T, m.H, rule=hot_side_4)

        def hot_side_5(m, t, h):                                            # Constraint defining q_12
            # return m.q_12[t,h] * m.u_b[t,h] == m.m_flow * m.Cp_w * (m.T2[t,h] - m.T1[t,h])
            return m.q_12[t,h] == m.m_flow * m.Cp_w * (m.T2[t,h] - m.T1[t,h])
        m.hot_side_con5 = pyo.Constraint(m.T, m.H, rule=hot_side_5)

        def cold_side_1(m, t, h):                                           # Constraint for condition T3 >= T_pcm
            w = 100000
            return m.T3[t,h] <= m.T_pcm[t,h] + m.u_34[t,h] * w
        m.cold_side_con1 = pyo.Constraint(m.T, m.H, rule=cold_side_1)

        def cold_side_2(m, t, h):                                           # Constraint for condition T3 <= T_pcm
                w = 100000
                return m.T3[t,h] >= m.T_pcm[t,h] - (1 - m.u_34[t,h]) * w
        m.cold_side_con2 = pyo.Constraint(m.T, m.H, rule=cold_side_2)

        def cold_side_3(m, t, h):                                           # T4 in between T3 and T_pcm
            return m.T4[t,h] >= m.T_pcm[t,h] * m.u_34[t,h] + (1 - m.u_34[t,h]) * m.T3[t,h]
        m.cold_side_con3 = pyo.Constraint(m.T, m.H, rule=cold_side_3)

        def cold_side_4(m, t, h):                                           # T4 in between T_pcm and T3
            return m.T4[t,h] <= m.T3[t,h] * m.u_34[t,h] + (1 - m.u_34[t,h]) * m.T_pcm[t,h]
        m.cold_side_con4 = pyo.Constraint(m.T, m.H, rule=cold_side_4)

        def cold_side_5(m, t, h):                                           # Constraint defining q_34
            # return m.q_34[t,h] * m.u_b[t,h] == m.m_flow * m.Cp_w * (m.T4[t,h] - m.T3[t,h])
            return m.q_34[t,h] == m.m_flow * m.Cp_w * (m.T4[t,h] - m.T3[t,h])
        m.cold_side_con5 = pyo.Constraint(m.T, m.H, rule=cold_side_5)

        def COP_LT(m, t):                                                   # Heat to electric power of HPs
            if sum(m.Q_cool[t,l].value for l in m.L) <= 0 and (m.T_outdoor[t].value <= 15):      # No cooling demand -> heating mode
                return m.p_HP[t,'LT'] * m.COP[t] == m.q_HP[t,'LT']
            else:   # Cooling mode
                return m.p_HP[t,'LT'] * m.COP[t] ==  - m.q_HP[t,'LT']
        m.COP_HP_con = pyo.Constraint(m.T, rule=COP_LT)

        def COP_HT(m, t):                                                 # Heat to electric power
            return m.p_HP[t,'HT'] * m.COP[t] == m.q_HP[t,'HT']
        m.COP_HP_2_con = pyo.Constraint(m.T, rule=COP_HT)

        def T2_LT_bounds(m, t):                                             # Limits in T2 for the LT network
            if sum(m.Q_cool[t,l].value for l in m.L) <= 0 and (m.T_outdoor[t].value <= 15):       # No cooling demand -> heating mode
                return m.T_min_sh, m.T2[t,'LT'], m.T_max_sh
            else:
                return m.T_min_cool, m.T2[t,'LT'], m.T_max_cool
        m.T2_LT_bounds_con = pyo.Constraint(m.T, rule=T2_LT_bounds)

        def T2_HT_bounds(m, t):                                             # Limits in T2 for the HT network
            if sum(m.Q_cool[t,l].value for l in m.L) <= 0 and (m.T_outdoor[t].value <= 15):       # No cooling demand -> heating mode
                return m.T_min_hw, m.T2[t,'HT'], m.T_max_hw
            else:
                T_max = max(m.T_max_sh.value, m.T_max_hw.value)
                T_min = max(m.T_min_sh.value, m.T_min_hw.value)
                return T_min, m.T2[t,'HT'], T_max
        m.T2_HT_bounds_con = pyo.Constraint(m.T, rule=T2_HT_bounds)

        def p_HP_bounds(m, t, h):                                           # B.C. of p_HP
            return 0, m.p_HP[t,h], m.P_HP_rated[h]
        m.p_HP_bounds_con = pyo.Constraint(m.T, m.H, rule=p_HP_bounds)

        def T1_bounds(m,t,h):
            return m.T_min[h], m.T1[t,h], m.T_max[h]
        m.T1_bounds_con = pyo.Constraint(m.T, m.H, rule=T1_bounds)

        def T3_bounds(m, t, h):
            return m.T_min[h], m.T3[t, h], m.T_max[h]
        m.T3_bounds_con = pyo.Constraint(m.T, m.H, rule=T3_bounds)

        def T4_bounds(m, t, h):
            return m.T_min[h], m.T4[t, h], m.T_max[h]
        m.T4_bounds_con = pyo.Constraint(m.T, m.H, rule=T4_bounds)

        def T_pcm_bounds(m, t, h):
            return m.T_min[h], m.T_pcm[t, h], m.T_max[h]
        m.T_pcm_bounds_con = pyo.Constraint(m.T, m.H, rule=T_pcm_bounds)

        # Necessary??
        def conT1_2(m,h):
            return m.T1[0, h] == m.T2[0, h]
        m.conT1_con1 = pyo.Constraint(m.H, rule=conT1_2)

        def conT4_1(m, h):
            return m.T4[0, h] == m.T3[0, h]
        m.conT4_con1 = pyo.Constraint(m.H, rule=conT4_1)


        # BESS
        def e_BESS_bounds(m, t):                                            # B.C. of e_BESS
            return m.BESS_Emin, m.e_BESS[t], m.BESS_Emax
        m.e_BESS_bounds_con = pyo.Constraint(m.T, rule=e_BESS_bounds)

        def p_bess_ch_boudaries(m, t):                                         # B.C. of p_BESS
            return 0, m.p_BESS_ch[t], m.BESS_Pmax
        m.p_bess_ch_bounds_con = pyo.Constraint(m.T, rule=p_bess_ch_boudaries)

        def p_bess_dch_boudaries(m, t):                                         # B.C. of p_BESS
            return m.BESS_Pmin, m.p_BESS_dch[t], 0
        m.p_bess_dch_bounds_con = pyo.Constraint(m.T, rule=p_bess_dch_boudaries)

        def bess_1(m, t):                                                   # charging/discharging of BESS
            if t > 0: # and t < len(m.T): # starts and finishes half charge
                return m.e_BESS[t] == m.e_BESS[t-1] + (m.p_BESS_ch[t] + m.p_BESS_dch[t]) * 1
            else:           
                return m.e_BESS[t] == m.e_BESS[71]
        m.bess_1_con = pyo.Constraint(m.T, rule=bess_1)

        def bess_ch_no_dch(m, t):                                           # BESS charging without discharging
            return m.p_BESS_ch[t] * m.u_BESS[t] == m.p_BESS_dch[t] * (1-m.u_BESS[t])
        m.bess_ch_no_dch_con = pyo.Constraint(m.T, rule=bess_ch_no_dch)

        def deactivate_bess(m, t):                                          # set p_BESS to zero all time
            return m.p_BESS_ch[t] + m.p_BESS_dch[t] == 0
        m.deactivate_bess_con = pyo.Constraint(m.T, rule=deactivate_bess)

        # EVs
        def e_ev_bounds(m, t, s, time=time):                                # State of Energy (SOE) constraints:
            time_dt = datetime.strptime(time[t], '%m/%d/%Y %H:%M:%S')
            if time_dt <= m.Tarr[s].value:
                return m.e_EV[t,s] == 0 
            elif time_dt > m.Tarr[s].value and time_dt < m.Tdep[s].value: 
                return 0, m.e_EV[t,s], m.EV_Emax[s] 
            else:
                return m.e_EV[t,s] == m.EV_Emax[s] * m.ev_soft[s]  
        m.e_ev_bounds_con = pyo.Constraint(m.T, m.S, rule=e_ev_bounds)

        def p_ev_bounds(m, t, s, time=time):                                # Charging power limitation
            time_dt = datetime.strptime(time[t], '%m/%d/%Y %H:%M:%S')
            if time_dt < (m.Tarr[s].value) or \
               time_dt > (m.Tdep[s].value + timedelta(hours=1)):
                return m.p_EV[t,s] == 0
            else:
                return m.EV_Pmin[s], m.p_EV[t,s], m.EV_Pmax[s]
        m.p_ev_bounds_con = pyo.Constraint(m.T, m.S, rule=p_ev_bounds)

        def EV_charging(m, t, s, delta=delta, time=time):                   # EV charging rules 
            time_dt = datetime.strptime(time[t], '%m/%d/%Y %H:%M:%S')
            time_dt_1 = datetime.strptime(time[t-1], '%m/%d/%Y %H:%M:%S')

            # if T_arr <= t <= T_arr + dt   then the extra charing time is considered in timestep t
            # if time_dt > m.Tarr[s].value and time_dt < (m.Tarr[s].value + timedelta(hours=1)):
            #     deltaT = time_dt - m.Tarr[s].value
            #     deltaT = deltaT.total_seconds() / 3600
            #     return m.e_EV[t,s] == m.e_EV[t-1,s] + m.p_EV[t,s] * deltaT
            
            # if T_dep <= t <= T_dep + dt   then the extra charing time is considered in timestep t-1
            if time_dt > m.Tdep[s].value and time_dt < (m.Tdep[s].value + timedelta(hours=1)):
                deltaT = m.Tdep[s].value - time_dt_1
                deltaT = deltaT.total_seconds() / 3600        # difference in hours

                return m.e_EV[t,s] == m.e_EV[t-1,s] + m.p_EV[t,s] * deltaT
            
            # elif time_dt > m.Tarr[s].value and time_dt <= m.Tdep[s].value:
            elif time_dt > m.Tarr[s].value and time_dt <= m.Tdep[s].value:
                return (m.e_EV[t,s] == m.e_EV[t-1,s] + m.p_EV[t,s] * 1)
            
            else:
                return pyo.Constraint.Skip
        m.EV_charging_rule = pyo.Constraint(m.T, m.S, rule=EV_charging)

        def ev_soft_limit(m,s):
            return m.SET_EV <= m.ev_soft[s]
        m.ev_soft_limit_con = pyo.Constraint(m.S, rule=ev_soft_limit)
        

        # Transformers:
        def trafo_limits(m, t):                                             # Power withdrawal <= trafo tech. capacity
            return sum(m.p_EV[t,s] for s in m.S) + sum(m.p_HP[t,h] for h in m.H) \
                 + sum(m.p_el[t,l] for l in m.L) - sum(m.p_PV[t,p] for p in m.P) \
                 + m.p_BESS_ch[t] + m.p_BESS_dch[t] <= 10000 #sum(m.trafo_limit[tr] for tr in m.Tr)
        m.trafo_limits_con = pyo.Constraint(m.T, rule=trafo_limits)


        # ----------------------------
        # CONSTRAINTS for cases of study: 
        # ----------------------------

        def p_total(m, t):
            return m.p_tot[t] == sum(m.p_EV[t,s] for s in m.S) + sum(m.p_HP[t,h] for h in m.H) \
                                 + sum(m.p_el[t,l] for l in m.L) - sum(m.p_PV[t,p] for p in m.P) \
                                 + m.p_BESS_ch[t] + m.p_BESS_dch[t]
        m.p_total_con = pyo.Constraint(m.T, rule=p_total)

        ## different models (model0 doesn't have peak constraints)

        def p_peak_model2(m, t):                                       # all time peak
            if t >= 24 and t <= 48:
                return m.p_peak >= m.p_tot[t]
            else:
                return pyo.Constraint.Skip
        m.model2_con = pyo.Constraint(m.T, rule=p_peak_model2)

        def p_peak_model3(m, t):                                       # morning peak only
            if t >= 24 and t <= 36 and m.peak_schedule[t].value:
                return m.p_peak >= m.p_tot[t]
            else:
                return pyo.Constraint.Skip
        m.model3_con = pyo.Constraint(m.T, rule=p_peak_model3)

        def p_peak_model4(m, t):                                       # evening peak only
            if t >= 36 and t <= 48 and m.peak_schedule[t].value:
                return m.p_peak >= m.p_tot[t]
            else:
                return pyo.Constraint.Skip
        m.model4_con = pyo.Constraint(m.T, rule=p_peak_model4)

        def p_peak_model5(m, t):                                       # morning and evening peak
            if t >= 24 and t <= 48 and m.peak_schedule[t].value:
                return m.p_peak >= m.p_tot[t]
            else:
                return pyo.Constraint.Skip
        m.model5_con = pyo.Constraint(m.T, rule=p_peak_model5)
    update_constraints(m, time, day, delta)

    return m
    
def create_datelist(start, end):
    start_date = datetime.strptime(start, '%d/%m/%Y %H:%M') + timedelta(days=1)
    end_date = datetime.strptime(end, '%d/%m/%Y %H:%M') - timedelta(days=1)
    day = start_date

    date_list = []
    while day <= end_date:
        date_list.append(day)
        day += timedelta(days=1)

    return date_list

def solve(m, min_soc_ev, bat_capacity, tes_capacity):
    def objective_case(m, case):
            del m.obj

            if case == 'base':
                m.obj = pyo.Objective(rule=(sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S) - sum(sum(m.p_HP[t,h] for t in m.T) for h in m.H)), sense=pyo.maximize)

            elif case == 'peak':
                m.obj = pyo.Objective(rule=(m.p_peak - 10**-6 * (sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S) - sum(sum(m.p_HP[t,h] for t in m.T) for h in m.H))), sense=pyo.minimize)

            elif case == 'evs':
                m.obj = pyo.Objective(rule=( m.p_peak - sum(m.ev_soft[s] * m.EV_Emax[s].value for s in m.S) \
                                     - 10**-6 * (sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S) - sum(sum(m.p_HP[t,h] for t in m.T) for h in m.H))), sense=pyo.minimize)
                            # No epsilons in the objective function???

            elif case == 'evs1':
                m.obj = pyo.Objective(rule=( m.p_peak - 10**-6 * sum(m.ev_soft[s] * m.EV_Emax[s].value for s in m.S) \
                                     - 10**-6 * (sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S) - sum(sum(m.p_HP[t,h] for t in m.T) for h in m.H))), sense=pyo.minimize)
                            # No epsilons in the objective function???

            elif case == 'bess' or case == 'bess v2g':        # add the SOC close to 70% thingy
                # m.obj = pyo.Objective(expr=(sum(m.p_peak[d] for d in m.peak_set) + sum(m.v_BESS[t] for t in m.T) \
                #                     - (10**-6 * sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S), sense=pyo.minimize)
                m.obj = pyo.Objective(expr=(m.p_peak - 10**-6 * ( sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S) - \
                                    sum(sum(m.p_HP[t,h] for t in m.T) for h in m.H) - sum((0.7*m.BESS_capacity - m.e_BESS[t])**2 for t in m.T) )), sense=pyo.minimize)

            elif case == 'v2g':
                m.obj = pyo.Objective(expr=(m.p_peak - 10**-6 * (sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S) - sum(sum(m.p_HP[t,h] for t in m.T) for h in m.H)) ), sense=pyo.minimize)
                # m.obj = pyo.Objective(expr=(m.p_peak - 10**-6 * (sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S) - sum(sum(m.p_HP[t,h] for t in m.T) for h in m.H) + sum(sum(m.e_EV[t,s] - m.e_EV[t-1,s] for s in m.S) for t in range(1,71))) ), sense=pyo.minimize)    

            elif case == 'tes':
                m.obj = pyo.Objective(expr=(m.p_peak - 10**-6 * (sum(sum(m.e_EV[t,s] for t in m.T) for s in m.S) - sum(sum(m.p_HP[t,h] for t in m.T) for h in m.H)) ), sense=pyo.minimize)

            else:
                print('------ ERROR: not valid objective ------')

    def constraints_case(m, case=''):

        if case == 'base':
            m.model2_con.deactivate()           # OFF
            m.model3_con.deactivate()           # OFF
            m.model4_con.deactivate()           # OFF
            m.model5_con.deactivate()           # OFF
            m.SET_EV = 1
            m.deactivate_bess_con.activate()    # ON
            for s in m.S:
                m.EV_Pmin[s] = 0
            
        elif case == 'peak':
            m.deactivate_bess_con.activate()    # ON
            m.SET_EV = 1
            for s in m.S:
                m.EV_Pmin[s] = 0

        elif case == 'evs' or case == 'evs1':
            m.deactivate_bess_con.activate()    # ON
            for s in m.S:
                m.EV_Pmin[s] = 0
            m.BESS_capacity = 500
            m.BESS_Pmax = int(0.5 * 500)
            m.BESS_Pmin = - int(0.5 * 500)
            m.BESS_Emax = int(0.85 * 500)
            m.BESS_Emin = int(0.15 * 500)

        elif case == 'bess' or case == 'bess v2g':
            m.deactivate_bess_con.deactivate()  # OFF
            m.SET_EV = 1
            # for s in m.S:
            #     m.EV_Pmin[s] = 0
            m.m_pcm['LT'] = 10000
            m.m_pcm['HT'] = 10000
            for h in m.H:
                E_pcm_0 = m.m_pcm[h].value * (273.15) * m.c_pcm[h].value / 3600               # [kWh]
                E_pcm_min = m.m_pcm[h].value * (273.15 + m.T_min[h].value) * m.c_pcm[h].value / 3600               # [kWh]
                m.E_pcm_min[h] = E_pcm_min - E_pcm_0
                E_s = m.m_pcm[h].value * (273.15 + m.T_m[h].value) * m.c_pcm[h].value / 3600                       # [kWh]
                m.E_s[h] = E_s - E_pcm_0
                E_l = E_s + m.m_pcm[h].value * m.h_pcm[h].value / 3600                                  # [kWh]
                m.E_l[h] = E_l - E_pcm_0
                E_pcm_max = m.m_pcm[h].value * ((273.15 + m.T_max[h].value) * m.c_pcm[h].value + m.h_pcm[h].value) / 3600     # [kWh]
                m.E_pcm_max[h] = E_pcm_max - E_pcm_0

        elif case == 'v2g':
            m.deactivate_bess_con.activate()    # ON
            m.SET_EV = 1
            m.BESS_capacity = 500
            m.BESS_Pmax = int(0.5 * 500)
            m.BESS_Pmin = - int(0.5 * 500)
            m.BESS_Emax = int(0.85 * 500)
            m.BESS_Emin = int(0.15 * 500)
        
        elif case == 'tes':
            m.deactivate_bess_con.deactivate()  # OFF
            m.SET_EV = 1
            for s in m.S:
                m.EV_Pmin[s] = 0
            m.BESS_capacity = 500
            m.BESS_Pmax = int(0.5 * 500)
            m.BESS_Pmin = - int(0.5 * 500)
            m.BESS_Emax = int(0.85 * 500)
            m.BESS_Emin = int(0.15 * 500)

    def store_results(m):
        results = {
            'name': m.name,
            
            'p_peak': m.p_peak.value,
            'p_tot_t': np.array([m.p_tot[t].value for t in m.T]),
            'schedule_t': np.array([m.peak_schedule[t].value for t in m.T]),

            'm.S': m.S.data(),
            'p_ev_ts': np.array([[m.p_EV[t,s].value for s in m.S] for t in m.T]),
            'e_ev_ts': np.array([[m.e_EV[t,s].value for s in m.S] for t in m.T]), 
            'ev_soft_s': np.array([m.ev_soft[s].value for s in m.S]),
            'E_ev_max_s': np.array([m.EV_Emax[s].value for s in m.S]),
            'P_ev_max_s': np.array([m.EV_Pmax[s].value for s in m.S]),
            'P_ev_min_s': np.array([m.EV_Pmin[s].value for s in m.S]),
            'T_arr_s': np.array([m.Tarr[s].value for s in m.S]),
            'T_dep_s': np.array([m.Tdep[s].value for s in m.S]),
            'set_ev': m.SET_EV.value,

            'E_bess_max': m.BESS_Emax.value,
            'E_bess_min': m.BESS_Emin.value,
            'E_bess_capacity': m.BESS_capacity.value,
            'e_bess_t': np.array([m.e_BESS[t].value for t in m.T]),
            'p_bess_ch_t': np.array([m.p_BESS_ch[t].value for t in m.T]),
            'p_bess_dch_t': np.array([m.p_BESS_dch[t].value for t in m.T]),

            'm.H': m.H.data(),           
            'p_hp_th': np.array([[m.p_HP[t,h].value for h in m.H] for t in m.T]),
            'q_hp_th': np.array([[m.q_HP[t,h].value for h in m.H] for t in m.T]),
            'e_pcm_th': np.array([[m.e_pcm[t,h].value for h in m.H] for t in m.T]),
            'q_12_th': np.array([[m.q_12[t,h].value for h in m.H] for t in m.T]),
            'q_34_th': np.array([[m.q_34[t,h].value for h in m.H] for t in m.T]),
            'T1_th': np.array([[m.T1[t,h].value for h in m.H] for t in m.T]),
            'T2_th': np.array([[m.T2[t,h].value for h in m.H] for t in m.T]),
            'T3_th': np.array([[m.T3[t,h].value for h in m.H] for t in m.T]),
            'T4_th': np.array([[m.T4[t,h].value for h in m.H] for t in m.T]),
            'T_pcm_th': np.array([[m.T_pcm[t,h].value for h in m.H] for t in m.T]),
            'u_12_th': np.array([[m.u_12[t,h].value for h in m.H] for t in m.T]),
            'u_34_th': np.array([[m.u_34[t,h].value for h in m.H] for t in m.T]),
            'u_l_th': np.array([[m.u_l[t,h].value for h in m.H] for t in m.T]),
            'u_s_th': np.array([[m.u_s[t,h].value for h in m.H] for t in m.T]),
            'm_pcm_h': np.array([m.m_pcm[h].value for h in m.H]),
            'E_max_h': np.array([m.E_pcm_max[h].value for h in m.H]),
            'E_min_h': np.array([m.E_pcm_min[h].value for h in m.H]),
            'E_s_h': np.array([m.E_s[h].value for h in m.H]),
            'E_l_h': np.array([m.E_l[h].value for h in m.H]),
            'T_min_h': np.array([m.T_min[h].value for h in m.H]),
            'T_max_h': np.array([m.T_max[h].value for h in m.H]),
            'COP_t': np.array([m.COP[t].value for t in m.T]),

            'm.L': m.L.data(),
            'p_el_tl': np.array([[m.p_el[t,l].value for l in m.L] for t in m.T]),
            'Q_sh_tl': np.array([[m.Q_sh[t,l].value for l in m.L] for t in m.T]),
            'Q_hw_tl': np.array([[m.Q_hw[t,l].value for l in m.L] for t in m.T]),
            'Q_cool_tl': np.array([[m.Q_cool[t,l].value for l in m.L] for t in m.T]),
            'N_houses_l': np.array([m.N_houses[l].value for l in m.L]),

            'm.P': m.P.data(),
            'p_pv_tp': np.array([[m.p_PV[t,p].value for p in m.P] for t in m.T]),
            }

        return results

    opt=pyo.SolverFactory('gurobi')

    # Scenario 1: Base case
    objective_case(m, 'base')
    constraints_case(m, 'base')
    solved = opt.solve(m)
    if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        m.name = 'base'
        scenario1 = store_results(m)
    else:
        print('------ ERROR: base case unfeasible ------')
        scenario1 = 0
        m.error = True
        infeasible_constraints = []
        for constr in m.component_data_objects(pyo.Constraint, active=True):
            if (constr.lower is not None and pyo.value(constr.lower) > 0) or \
            (constr.upper is not None and pyo.value(constr.upper) < 0):
                infeasible_constraints.append(constr)
        
        # Print or process information about the infeasible constraints
        for constr in infeasible_constraints:
            print("Constraint '{}' is violated".format(constr.name))

    list_cases = ['bess'] #['peak', 'evs', 'evs1', 'v2g', 'tes', 'bess'] #

    scenario2 = {
        'peak': {},
        'evs': {},
        'bess': {},
        'v2g': {},
        'tes': {},
        'evs1': {},
        'bess v2g': {}
    }
    scenario3 = {
        'peak': {},
        'evs': {},
        'bess': {},
        'v2g': {},
        'tes': {},
        'evs1': {},
        'bess v2g': {}
    }
    scenario4 = {
        'peak': {},
        'evs': {},
        'bess': {},
        'v2g': {},
        'tes': {},
        'evs1': {},
        'bess v2g': {}
    }
    scenario5 = {
        'peak': {},
        'evs': {},
        'bess': {},
        'v2g': {},
        'tes': {},
        'evs1': {},
        'bess v2g': {}
    }

    for case in list_cases:

        # Scenario 2: all time peak
        objective_case(m, case)
        constraints_case(m, case)
        m.model2_con.activate()
        m.model3_con.deactivate()
        m.model4_con.deactivate()
        m.model5_con.deactivate()
        if case == 'peak':
            solved = opt.solve(m)
            if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
                m.name = f'peak'
                scenario2[case] = store_results(m)
            else:
                print(f'------ ERROR: peak (scenario 2) unfeasible ------')
                scenario2[case]  = 0
                m.error = True

        elif case == 'evs':
            m.SET_EV = min_soc_ev*0.01
            solved = opt.solve(m)
            if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
                m.name = f'evs {min_soc_ev}'
                scenario2[case] = store_results(m)
            else:
                    print(f'------ ERROR: evs - {min_soc_ev} (scenario 2) unfeasible ------')
                    scenario2[case] = 0
                    m.error = True

        elif case == 'evs1':
            m.SET_EV = min_soc_ev*0.01
            solved = opt.solve(m)
            if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
                m.name = f'evs1 {min_soc_ev}'
                scenario2[case] = store_results(m)
            else:
                    print(f'------ ERROR: evs - {min_soc_ev} (scenario 3) unfeasible ------')
                    scenario3[case] = 0
                    m.error = True

        elif case == 'tes':
            for e in tes_capacity:   # changing the capacity of TES only by increasing the mass
                m.m_pcm['LT'] = e
                m.m_pcm['HT'] = e
                for h in m.H:
                    # h = 'HT'
                    E_pcm_0 = m.m_pcm[h].value * (273.15) * m.c_pcm[h].value / 3600               # [kWh]
                    E_pcm_min = m.m_pcm[h].value * (273.15 + m.T_min[h].value) * m.c_pcm[h].value / 3600               # [kWh]
                    m.E_pcm_min[h] = E_pcm_min - E_pcm_0
                    E_s = m.m_pcm[h].value * (273.15 + m.T_m[h].value) * m.c_pcm[h].value / 3600                       # [kWh]
                    m.E_s[h] = E_s - E_pcm_0
                    E_l = E_s + m.m_pcm[h].value * m.h_pcm[h].value / 3600                                  # [kWh]
                    m.E_l[h] = E_l - E_pcm_0
                    E_pcm_max = m.m_pcm[h].value * ((273.15 + m.T_max[h].value) * m.c_pcm[h].value + m.h_pcm[h].value) / 3600     # [kWh]
                    m.E_pcm_max[h] = E_pcm_max - E_pcm_0
                solved = opt.solve(m)
                if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
                    m.name = f'tes {e}'
                    scenario2[case][e] = store_results(m)
                else:
                    print(f'------ ERROR: tes {e} (scenario 2) unfeasible ------')
                    scenario2[case][e] = 0
                    m.error = True

        elif case == 'bess':
            for b in bat_capacity:
                for c_rate in [0.5, 1]:
                    m.BESS_capacity = b
                    m.BESS_Pmax = int(c_rate * b)
                    m.BESS_Pmin = - int(c_rate * b)
                    m.BESS_Emax = int(0.85 * b)
                    m.BESS_Emin = int(0.15 * b)
                    opt.options['TimeLimit'] = 600
                    solved = opt.solve(m)
                    if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
                        m.name = f'bess {b} - {c_rate}C'
                        skip = False
                        scenario2[case][(b,c_rate)] = store_results(m)
                        # print(day, b, c_rate, 'compiled for scenario 2')
                    else:
                        print(f'------ ERROR: bess - {b} | {c_rate} (scenario 2) unfeasible ------')
                        scenario2[case][(b,c_rate)]  = 0
                        m.error = True
                        skip = True
                #     if skip:
                #         break
                # if skip:
                #     break

        elif case == 'v2g':
            for s in m.S:
                m.EV_Pmin[s] = - m.EV_Pmax[s].value
            solved = opt.solve(m)
            if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
                m.name = f'v2g'
                scenario2[case] = store_results(m)
            else:
                print(f'------ ERROR: v2g (scenario 2) unfeasible ------')
                scenario2[case]  = 0
                m.error = True
        
        elif case == 'bess v2g':
            for b in bat_capacity:
                c_rate = 0.5
                m.BESS_capacity = b
                m.BESS_Pmax = int(c_rate * b)
                m.BESS_Pmin = - int(c_rate * b)
                m.BESS_Emax = int(0.85 * b)
                m.BESS_Emin = int(0.15 * b)
                for s in m.S:
                    m.EV_Pmin[s] = - m.EV_Pmax[s].value
                solved = opt.solve(m)
                if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
                    m.name = f'bess {b} - {c_rate}C'
                    scenario2[case][(b,c_rate)] = store_results(m)
                else:
                    print(f'------ ERROR: bess - {b} | {c_rate} (scenario 2) unfeasible ------')
                    scenario2[case][(b,c_rate)]  = 0
                    m.error = True
        else:
            print('------ ERROR: not valid scenario 2 ------')
            scenario2 = 0
            m.error = True
            

        # if skip:
        #     print('skipping', time.time())
        #     continue    
        # Scenario 3: morning peak only
        # objective_case(m, case)
        # constraints_case(m, case)
        # m.model2_con.deactivate()
        # m.model3_con.activate()
        # m.model4_con.deactivate()
        # m.model5_con.deactivate()
        # solved = opt.solve(m)
        # if case == 'peak':
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'peak'
        #         scenario3[case] = store_results(m)
        #     else:
        #         print(f'------ ERROR: peak (scenario 3) unfeasible ------')
        #         scenario3[case]  = 0
        #         m.error = True

        # elif case == 'evs':
        #     m.SET_EV = min_soc_ev*0.01
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'evs {min_soc_ev}'
        #         scenario3[case] = store_results(m)
        #     else:
        #             print(f'------ ERROR: evs - {min_soc_ev} (scenario 3) unfeasible ------')
        #             scenario3[case] = 0
        #             m.error = True

        # elif case == 'evs1':
        #     m.SET_EV = min_soc_ev*0.01
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'evs1 {min_soc_ev}'
        #         scenario3[case] = store_results(m)
        #     else:
        #             print(f'------ ERROR: evs - {min_soc_ev} (scenario 3) unfeasible ------')
        #             scenario3[case] = 0
        #             m.error = True

        # elif case == 'tes':
        #     for e in tes_capacity:   # changing the capacity of TES only by increasing the mass
        #         m.m_pcm['LT'] = e
        #         # m.m_pcm['HT'] = e
        #         for h in m.H:
        #             h = 'HT'
        #             E_pcm_0 = m.m_pcm[h].value * (273.15) * m.c_pcm[h].value / 3600               # [kWh]
        #             E_pcm_min = m.m_pcm[h].value * (273.15 + m.T_min[h].value) * m.c_pcm[h].value / 3600               # [kWh]
        #             m.E_pcm_min[h] = E_pcm_min - E_pcm_0
        #             E_s = m.m_pcm[h].value * (273.15 + m.T_m[h].value) * m.c_pcm[h].value / 3600                       # [kWh]
        #             m.E_s[h] = E_s - E_pcm_0
        #             E_l = E_s + m.m_pcm[h].value * m.h_pcm[h].value / 3600                                  # [kWh]
        #             m.E_l[h] = E_l - E_pcm_0
        #             E_pcm_max = m.m_pcm[h].value * ((273.15 + m.T_max[h].value) * m.c_pcm[h].value + m.h_pcm[h].value) / 3600     # [kWh]
        #             m.E_pcm_max[h] = E_pcm_max - E_pcm_0
        #         solved = opt.solve(m)
        #         if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #             m.name = f'tes {e}'
        #             scenario3[case][e] = store_results(m)
        #         else:
        #             print(f'------ ERROR: tes {e} (scenario 3) unfeasible ------')
        #             scenario3[case][e] = 0
        #             m.error = True

        # elif case == 'bess':
        #     for b in bat_capacity:
        #         for c_rate in [0.5, 1]:
        #             m.BESS_capacity = b
        #             m.BESS_Pmax = int(c_rate * b)
        #             m.BESS_Pmin = - int(c_rate * b)
        #             m.BESS_Emax = int(0.85 * b)
        #             m.BESS_Emin = int(0.15 * b)
        #             solved = opt.solve(m)
        #             opt.options['TimeLimit'] = 60
        #             if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #                 m.name = f'bess {b} - {c_rate}C'
        #                 scenario3[case][(b,c_rate)] = store_results(m)
        #             else:
        #                 print(f'------ ERROR: bess - {b} | {c_rate} (scenario 3) unfeasible ------')
        #                 scenario3[case][(b,c_rate)]  = 0
        #                 m.error = True

        # elif case == 'v2g':
        #     for s in m.S:
        #         m.EV_Pmin[s] = - m.EV_Pmax[s].value
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'v2g'
        #         scenario3[case] = store_results(m)
        #     else:
        #         print(f'------ ERROR: v2g (scenario 3) unfeasible ------')
        #         scenario3[case]  = 0
        #         m.error = True
        
        # elif case == 'bess v2g':
        #     for b in bat_capacity:
        #         c_rate = 0.5
        #         m.BESS_capacity = b
        #         m.BESS_Pmax = int(c_rate * b)
        #         m.BESS_Pmin = - int(c_rate * b)
        #         m.BESS_Emax = int(0.85 * b)
        #         m.BESS_Emin = int(0.15 * b)
        #         for s in m.S:
        #             m.EV_Pmin[s] = - m.EV_Pmax[s].value
        #         solved = opt.solve(m)
        #         if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #             m.name = f'bess {b} - {c_rate}C'
        #             scenario3[case][(b,c_rate)] = store_results(m)
        #         else:
        #             print(f'------ ERROR: bess - {b} | {c_rate} (scenario 3) unfeasible ------')
        #             scenario3[case][(b,c_rate)]  = 0
        #             m.error = True


        # else:
        #     print('------ ERROR: not valid scenario 3 ------')
        #     scenario3 = 0
        #     m.error = True
            
        # Scenario 4: evening peak only
        # objective_case(m, case)
        # constraints_case(m, case)
        # m.model2_con.deactivate()
        # m.model3_con.deactivate()
        # m.model4_con.activate()
        # m.model5_con.deactivate()
        # solved = opt.solve(m)
        # if case == 'peak':
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'peak'
        #         scenario4[case] = store_results(m)
        #     else:
        #         print(f'------ ERROR: peak (scenario 4) unfeasible ------')
        #         scenario4[case]  = 0
        #         m.error = True

        # elif case == 'evs':
        #     m.SET_EV = min_soc_ev*0.01
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'evs {min_soc_ev}'
        #         scenario4[case] = store_results(m)
        #     else:
        #             print(f'------ ERROR: evs - {min_soc_ev} (scenario 4) unfeasible ------')
        #             scenario4[case] = 0
        #             m.error = True
        
        # elif case == 'evs1':
        #     m.SET_EV = min_soc_ev*0.01
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'evs1 {min_soc_ev}'
        #         scenario4[case] = store_results(m)
        #     else:
        #             print(f'------ ERROR: evs - {min_soc_ev} (scenario 3) unfeasible ------')
        #             scenario3[case] = 0
        #             m.error = True

        # elif case == 'tes':
        #     for e in tes_capacity:   # changing the capacity of TES only by increasing the mass
        #         m.m_pcm['LT'] = e
        #         m.m_pcm['HT'] = e
        #         for h in m.H:
        #             # h = 'HT'
        #             E_pcm_0 = m.m_pcm[h].value * (273.15) * m.c_pcm[h].value / 3600               # [kWh]
        #             E_pcm_min = m.m_pcm[h].value * (273.15 + m.T_min[h].value) * m.c_pcm[h].value / 3600               # [kWh]
        #             m.E_pcm_min[h] = E_pcm_min - E_pcm_0
        #             E_s = m.m_pcm[h].value * (273.15 + m.T_m[h].value) * m.c_pcm[h].value / 3600                       # [kWh]
        #             m.E_s[h] = E_s - E_pcm_0
        #             E_l = E_s + m.m_pcm[h].value * m.h_pcm[h].value / 3600                                  # [kWh]
        #             m.E_l[h] = E_l - E_pcm_0
        #             E_pcm_max = m.m_pcm[h].value * ((273.15 + m.T_max[h].value) * m.c_pcm[h].value + m.h_pcm[h].value) / 3600     # [kWh]
        #             m.E_pcm_max[h] = E_pcm_max - E_pcm_0
        #         solved = opt.solve(m)
        #         if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #             m.name = f'tes {e}'
        #             scenario4[case][e] = store_results(m)
        #         else:
        #             print(f'------ ERROR: tes {e} (scenario 4) unfeasible ------')
        #             scenario4[case][e] = 0
        #             m.error = True

        # elif case == 'bess':
        #     for b in bat_capacity:
        #         for c_rate in [0.5, 1]:
        #             m.BESS_capacity = b
        #             m.BESS_Pmax = int(c_rate * b)
        #             m.BESS_Pmin = - int(c_rate * b)
        #             m.BESS_Emax = int(0.85 * b)
        #             m.BESS_Emin = int(0.15 * b)
        #             solved = opt.solve(m)
        #             opt.options['TimeLimit'] = 60
        #             if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #                 m.name = f'bess {b} - {c_rate}C'
        #                 scenario4[case][(b,c_rate)] = store_results(m)
        #             else:
        #                 print(f'------ ERROR: bess - {b} | {c_rate} (scenario 4) unfeasible ------')
        #                 scenario4[case][(b,c_rate)]  = 0
        #                 m.error = True

        # elif case == 'v2g':
        #     for s in m.S:
        #         m.EV_Pmin[s] = - m.EV_Pmax[s].value
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'v2g'
        #         scenario4[case] = store_results(m)
        #     else:
        #         print(f'------ ERROR: v2g (scenario 4) unfeasible ------')
        #         scenario4[case]  = 0
        #         m.error = True
        
        # elif case == 'bess v2g':
        #     for b in bat_capacity:
        #         c_rate = 0.5
        #         m.BESS_capacity = b
        #         m.BESS_Pmax = int(c_rate * b)
        #         m.BESS_Pmin = - int(c_rate * b)
        #         m.BESS_Emax = int(0.85 * b)
        #         m.BESS_Emin = int(0.15 * b)
        #         for s in m.S:
        #             m.EV_Pmin[s] = - m.EV_Pmax[s].value
        #         solved = opt.solve(m)
        #         if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #             m.name = f'bess {b} - {c_rate}C'
        #             scenario4[case][(b,c_rate)] = store_results(m)
        #         else:
        #             print(f'------ ERROR: bess - {b} | {c_rate} (scenario 4) unfeasible ------')
        #             scenario4[case][(b,c_rate)]  = 0
        #             m.error = True

        # else:
        #     print('------ ERROR: not valid scenario 4 ------')
        #     scenario4 = 0
        #     m.error = True
            
        # Scenario 5: morning and evening peak
        # objective_case(m, case)
        # constraints_case(m, case)
        # m.model2_con.deactivate()
        # m.model3_con.deactivate()
        # m.model4_con.deactivate()
        # m.model5_con.activate()
        # solved = opt.solve(m)
        # if case == 'peak':
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'peak'
        #         scenario5[case] = store_results(m)
        #     else:
        #         print(f'------ ERROR: peak (scenario 5) unfeasible ------')
        #         scenario5[case]  = 0
        #         m.error = True

        # elif case == 'evs':
        #     m.SET_EV = min_soc_ev*0.01
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'evs {min_soc_ev}'
        #         scenario5[case] = store_results(m)
        #     else:
        #             print(f'------ ERROR: evs - {min_soc_ev} (scenario 5) unfeasible ------')
        #             scenario5[case] = 0
        #             m.error = True

        # elif case == 'evs1':
        #     m.SET_EV = min_soc_ev*0.01
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'evs1 {min_soc_ev}'
        #         scenario5[case] = store_results(m)
        #     else:
        #             print(f'------ ERROR: evs - {min_soc_ev} (scenario 3) unfeasible ------')
        #             scenario3[case] = 0
        #             m.error = True

        # elif case == 'tes':
        #     for e in tes_capacity:   # changing the capacity of TES only by increasing the mass
        #         m.m_pcm['LT'] = e
        #         m.m_pcm['HT'] = e
        #         for h in m.H:
        #             # h = 'HT'
        #             E_pcm_0 = m.m_pcm[h].value * (273.15) * m.c_pcm[h].value / 3600               # [kWh]
        #             E_pcm_min = m.m_pcm[h].value * (273.15 + m.T_min[h].value) * m.c_pcm[h].value / 3600               # [kWh]
        #             m.E_pcm_min[h] = E_pcm_min - E_pcm_0
        #             E_s = m.m_pcm[h].value * (273.15 + m.T_m[h].value) * m.c_pcm[h].value / 3600                       # [kWh]
        #             m.E_s[h] = E_s - E_pcm_0
        #             E_l = E_s + m.m_pcm[h].value * m.h_pcm[h].value / 3600                                  # [kWh]
        #             m.E_l[h] = E_l - E_pcm_0
        #             E_pcm_max = m.m_pcm[h].value * ((273.15 + m.T_max[h].value) * m.c_pcm[h].value + m.h_pcm[h].value) / 3600     # [kWh]
        #             m.E_pcm_max[h] = E_pcm_max - E_pcm_0
        #         solved = opt.solve(m)
        #         if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #             m.name = f'tes {e}'
        #             scenario5[case][e] = store_results(m)
        #         else:
        #             print(f'------ ERROR: tes {e} (scenario 5) unfeasible ------')
        #             scenario5[case][e] = 0
        #             m.error = True

        # elif case == 'bess':
        #     for b in bat_capacity:
        #         for c_rate in [0.5, 1]:
        #             m.BESS_capacity = b
        #             m.BESS_Pmax = int(c_rate * b)
        #             m.BESS_Pmin = - int(c_rate * b)
        #             m.BESS_Emax = int(0.85 * b)
        #             m.BESS_Emin = int(0.15 * b)
        #             solved = opt.solve(m)
        #             opt.options['TimeLimit'] = 60
        #             if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #                 m.name = f'bess {b} - {c_rate}C'
        #                 scenario5[case][(b,c_rate)] = store_results(m)
        #             else:
        #                 print(f'------ ERROR: bess - {b} | {c_rate} (scenario 5) unfeasible ------')
        #                 scenario5[case][(b,c_rate)]  = 0
        #                 m.error = True

        # elif case == 'v2g':
        #     for s in m.S:
        #         m.EV_Pmin[s] = - m.EV_Pmax[s].value
        #     solved = opt.solve(m)
        #     if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #         m.name = f'v2g'
        #         scenario5[case] = store_results(m)
        #     else:
        #         print(f'------ ERROR: v2g (scenario 5) unfeasible ------')
        #         scenario5[case]  = 0
        #         m.error = True
        
        # elif case == 'bess v2g':
        #     for b in bat_capacity:
        #         c_rate = 0.5
        #         m.BESS_capacity = b
        #         m.BESS_Pmax = int(c_rate * b)
        #         m.BESS_Pmin = - int(c_rate * b)
        #         m.BESS_Emax = int(0.85 * b)
        #         m.BESS_Emin = int(0.15 * b)
        #         for s in m.S:
        #             m.EV_Pmin[s] = - m.EV_Pmax[s].value
        #         solved = opt.solve(m)
        #         if solved.solver.status == pyo.SolverStatus.ok and solved.solver.termination_condition == pyo.TerminationCondition.optimal:
        #             m.name = f'bess {b} - {c_rate}C'
        #             scenario5[case][(b,c_rate)] = store_results(m)
        #         else:
        #             print(f'------ ERROR: bess - {b} | {c_rate} (scenario 5) unfeasible ------')
        #             scenario5[case][(b,c_rate)]  = 0
        #             m.error = True

        # else:
        #     print('------ ERROR: not valid scenario 5 ------')
        #     scenario5 = 0
        #     m.error = True

        # if solved.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
        #     print("The solver exceeded the time limit.")

    return {'scenario2': scenario2, 'scenario3': scenario3, 'scenario4': scenario4, 'scenario5': scenario5, 'scenario1': scenario1}
        
# ------------------------------------------------------------------------

filename = 'Data V2.xlsx'

abs_model = create_model()

# start_date = '31/12/2020 00:00'      # start date in the format: 'DD/MM/YYYY HH:MM'
# end_date = '31/01/2021 23:59'        # end date in the format:   'DD/MM/YYYY HH:MM'

# date_list = create_datelist(start_date, end_date)

# date_list = [create_datelist('24/01/2021 00:00', '26/01/2021 23:59'),\
#              create_datelist('03/05/2021 00:00', '05/05/2021 23:59'),\
#             create_datelist('08/07/2021 00:00', '10/07/2021 23:59'),\
#             create_datelist('12/10/2021 00:00', '14/10/2021 23:59')]

data = read_input_file(filename,'processed_inputd.pkl', re_run=False)
print('--- Data imported ---') 

# Set the cases and sensitivity analysis:
min_soc_ev = 70 # [%]
# bat_capacity = range(1000, 8000, 1000)   # [kWh]
bat_capacity = range(500,3000, 500) #[300, 500, 700, 1100, 1300, 1500, 1700, 1900, 2200] # range(2000, 14000, 2000)   # [kWh]
tes_capacity = range(10000, 200000, 20000) #20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000[] # range(5000, 124073, 10000)   # [kWh]
# ---------------------------

start_time = time.time()
results = {}
scenario1 = {}
scenario2 = {}
scenario3 = {}
scenario4 = {}
scenario5 = {}


# --------------------------
dt_object = datetime.fromtimestamp(time.time())
time_string = dt_object.strftime("%H:%M")
print('---- Start', '----', time_string)

from tqdm import tqdm
# for day in tqdm(date_list):
for day_str in ['23-12-2021 00:00']: #tqdm(['25-01-2021 00:00', '04-05-2021 00:00', '09-07-2021 00:00', '13-10-2021 00:00']):

    # day_str = day.strftime('%d-%m-%Y')

    day = datetime.strptime(day_str, '%d-%m-%Y %H:%M')

    m = initialize_model(abs_model, data, day, 'Hours')

    results[day] = solve(m, min_soc_ev, bat_capacity, tes_capacity)

    # scenario1[day_str] = results[day]['scenario1']
    scenario2[day_str] = results[day]['scenario2']
    # scenario3[day_str] = results[day]['scenario3']
    # scenario4[day_str] = results[day]['scenario4']
    # scenario5[day_str] = results[day]['scenario5']

    # Ensure the directory exists
    # directory = 'scenario1_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # file_path = os.path.join(directory, f'{day_str[0:10]}_tes_to.pkl')
    # with open(file_path, 'wb') as file:
    #     pickle.dump(, file)

    # directory = 'scenario2_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # file_path = os.path.join(directory, f'{day_str[0:10]}_tes_to.pkl')
    # with open(file_path, 'wb') as file:
    #     pickle.dump(scenario2[day_str[0:10]], file)

    # directory = 'scenario3_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # file_path = os.path.join(directory, f'{day_str[0:10]}_tes_to.pkl')
    # with open(file_path, 'wb') as file:
    #     pickle.dump(scenario3[day_str[0:10]], file)

    # directory = 'scenario4_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # file_path = os.path.join(directory, f'{day_str[0:10]}_tes_to.pkl')
    # with open(file_path, 'wb') as file:
    #     pickle.dump(scenario4[day_str[0:10]], file)

    # directory = 'scenario5_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # file_path = os.path.join(directory, f'{day_str[0:10]}_tes_to.pkl')
    # with open(file_path, 'wb') as file:
    #     pickle.dump(scenario5[day_str[0:10]], file)

    # directory = 'scenario1_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # pickle.dump(scenario1[day_str], open(f'scenario1_to/{day_str[0:10]}_bess.pkl', 'wb'))
    directory = 'scenario2_to'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(scenario2[day_str], open(f'scenario2_to/{day_str[0:10]}_max.pkl', 'wb'))
    # directory = 'scenario3_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # pickle.dump(scenario3[day_str], open(f'scenario3_to/{day_str[0:10]}_bess.pkl', 'wb'))
    # directory = 'scenario4_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # pickle.dump(scenario4[day_str], open(f'scenario4_to/{day_str[0:10]}_bess.pkl', 'wb'))
    # directory = 'scenario5_to'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # pickle.dump(scenario5[day_str], open(f'scenario5_to/{day_str[0:10]}_bess.pkl', 'wb'))


dt_object = datetime.fromtimestamp(time.time())
time_string = dt_object.strftime("%H:%M")
print('--- Finish ---', time_string)


# Saving the results:


# pickle.dump(scenario1, open('scenario1.pkl', 'wb'))
# pickle.dump(scenario2, open('scenario2.pkl', 'wb'))
# pickle.dump(scenario3, open('scenario3.pkl', 'wb'))
# pickle.dump(scenario4, open('scenario4.pkl', 'wb'))
# pickle.dump(scenario5, open('scenario5.pkl', 'wb'))
