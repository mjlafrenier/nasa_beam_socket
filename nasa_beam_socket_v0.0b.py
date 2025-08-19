# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:05:44 2025

@author: michael.lafrenier
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import  tkinter as tk
from tkinter import filedialog
import tkinter.messagebox
import os
import tqdm
import time

start_time = time.time()

def get_user_input():

    root = tk.Tk()
    root.withdraw()
    
    print("Select xs input csv file...")
    input_filename = filedialog.askopenfilename(title = "Select input csv file:")
    oDir = os.path.dirname(input_filename) + "//"
    
    # READ EID INPUT DATA TO DATAFRAME
    # NOTE: ISSUE WITH READING IN "NA", PANDAS(NUMPY) CONSIDER THAT AS A DEFAULT nan VALUE, THEREFORE
    # WE CAN EITHER TOSS DEFAULT NA VALUES USING pd.read_csv "keep_default_na=False, na_values=['']" OR JUST NOT USE "NA"
    # AS AN INPUT
    user_input_df = pd.read_csv(input_filename)
    user_input_df = user_input_df.reset_index()
    
    return input_filename, oDir, user_input_df

def NASA_beam_socket(ID,SC,Fo, Mo, L, a, b,n):
    '''
    Script to calculate beam socket analysis for continuous contact.
    
    Source:
    Rash, Larry C. "Strength Evaluation of Socket Joints". NASA Contractor Report 4608. June 1994
    

    Fwd joint member - pin
    AFt joint member - socket
    
    '''
    # FWD JOINT MEMBER WITH CONTINUOUS CONTACT
    
    c = ((3*(b-a)*(Mo + Fo*(L-a)) - (2*((b-a)**2)*Fo)) / (6*(Mo + Fo*(L-a)) - (3*(b-a)*Fo))) # LOAD REVERSAL LOCATION
    c_o = L - c - a
    
    W1 = (3*(Mo + Fo*(L-a)) - (c*Fo)) / (2*(b-a)) # CONCENTRATED LOAD
    W2 = (3*(Mo + Fo*(L-a)) - ((c+(2*(b-a)))*Fo)) / (2*(b-a)) # CONCENTRATED LOAD    
    
    # Geometric simplifications
    BAC = b - a - c
    
    # MOMENT CALCULATIONS
    # Initialize lists
    x_lst = []
    Mx1_lst = []
    Mx2_lst = []
    loc_lst = []
    x = L - b
    # Subscript 1 is pin, subscript 2 is socket
    for i in range(0,n+1):
        XLB = x - L + b
        x_lst.append(x)
        if x >= (L-b) and x < (L-a-c):
            Mx1_lst.append(Mo + Fo*x - ((W1/3)*(((3*XLB**2)/(BAC))-((XLB**3)/(BAC**2)))))
            Mx2_lst.append(            ((W1/3)*(((3*XLB**2)/(BAC))-((XLB**3)/(BAC**2)))))
            loc_lst.append('fwd')
        elif x >= (L-a-c) and x <= ((L-a)+0.01):
            Mx1_lst.append(Mo + Fo*x - (W1)*((XLB)-(BAC/3))+(W2*(((x-L+a+c)**3)/(3*c**2))))
            Mx2_lst.append(            (W1)*((XLB)-(BAC/3))-(W2*(((x-L+a+c)**3)/(3*c**2))))
            loc_lst.append('aft')
        else:
            Mx1_lst.append('error')
            Mx2_lst.append('error')
            
        if Mx1_lst[i] < 0.001:
            Mx1_lst[i] = 0.00
        else:
            pass
        
        if Mx2_lst[i] < 0.001:
            Mx2_lst[i] = 0.00
        else:
            pass
        
        x = x + (b-a)/(n)   
    
    Vx1_lst = []
    Vx2_lst = []
    # SHEAR CALCULATIONS
    for i in range(len(x_lst)):
        #print(i)
        if i == 0: # if first value
            Vx1_lst.append(Fo) # Shear is the slope of the moment at a given x
            Vx2_lst.append(0.0) # Shear is the slope of the moment at a given x
            #print(i)
        else:
            Vx1_lst.append((Mx1_lst[i]-Mx1_lst[i-1])/(x_lst[i]-x_lst[i-1])) # Shear is the slope of the moment at a given x
            Vx2_lst.append((Mx2_lst[i]-Mx2_lst[i-1])/(x_lst[i]-x_lst[i-1])) # Shear is the slope of the moment at a given x, starting with the negative shear input from the pin
            #print(i)
    
    # Add initial point for shear moment on pin
    # Subscript 1 - pin
    x_lst1 = x_lst.copy()
    x_lst1.insert(0, 0)
    loc_lst1 = loc_lst.copy()
    loc_lst1.insert(0, 'pin')
    Vx1_lst.insert(0, Fo)
    Mx1_lst.insert(0, Mo)
    
    # Subscript 2 - socket
    #Vx2_lst.insert(0, 0)
    #Mx2_lst.insert(0, 0)
    
    # Assembling analysis dataframe 
    df = pd.DataFrame(zip(x_lst1,loc_lst1,Vx1_lst,Mx1_lst), columns=['x','loc','Vx_pin','Mx_pin'])
    df2 = pd.DataFrame(zip(x_lst,loc_lst,Vx2_lst,Mx2_lst), columns=['x','loc','Vx_socket','Mx_socket'])
    
    df_final = pd.merge(df, df2, on=['x','loc'], how='left')
    df_final['ID'] = ID
    df_final['SC'] = SC
    df_final['L [in]'] = L
    df_final['a [in]'] = a
    df_final['b [in]'] = b
    df_final['n'] = n
    df_final['Fo [lb]'] = Fo
    df_final['Mo [in-lb]'] = Mo
    df_final['W1 [lb]'] = W1
    df_final['W2 [lb]'] = W2
    df_final['c [in]'] = c
    df_final = df_final[['ID','SC','L [in]','a [in]','b [in]','n','Fo [lb]', 'Mo [in-lb]','W1 [lb]','W2 [lb]','c [in]','x','loc','Vx_pin','Mx_pin','Vx_socket','Mx_socket']]
    
    # List of tuples for plotting
    plt_Mx1_lst = list(zip(x_lst1,Mx1_lst))
    plt_Vx1_lst = list(zip(x_lst1,Vx1_lst))
    plt_Mx2_lst = list(zip(x_lst,Mx2_lst))
    plt_Vx2_lst = list(zip(x_lst,Vx2_lst))
    
    return df_final, c_o, plt_Vx1_lst, plt_Mx1_lst, plt_Vx2_lst, plt_Mx2_lst

def hollow_cylinder_properties(Di, t):
    outer_radius = Di/2 + t
    inner_radius = Di/2
    
    # Calculate Cross-Sectional Area
    outer_area = math.pi * outer_radius ** 2
    inner_area = math.pi * inner_radius ** 2
    A = outer_area - inner_area

    # Calculate Cross-Sectional Moment of Inertia
    I = (math.pi/64)*(((Di+(2*t))**4)-((Di)**4))

    return A, I

def solid_cylinder_properties(D):
    # Calculate Cross-Sectional Area
    A = math.pi * ((D**2)/4)

    # Calculate Cross-Sectional Moment of Inertia
    I = (math.pi/64) * D**4

    return A, I

def plot_xs_pts(pts,pts1,pts2,pts3,c):
    # Plotting the generated points for visualization
    x, y = zip(*pts)
    x1, y1 = zip(*pts1)
    x2, y2 = zip(*pts2)
    x3, y3 = zip(*pts3)
    x4, y4 = zip(*c)
    
    plt.figure()
    plt.plot(x, y, 'b', label='Vx_pin')
    plt.plot(x1, y1, 'r', label='Mx_pin')
    plt.plot(x2, y2, 'b--', label='Vx_socket')
    plt.plot(x3, y3, 'r--', label='Mx_socket')
    plt.plot(x4, y4, 'bo', label='c_load_reversal')
    
    #plt.fill(x, y, 'b', alpha=0.1)
    
    # Label the points
    #for i in range(len(x)):
    #    plt.text(x[i], y[i], i, ha='center', va='bottom')
    xmin = 0.0
    xmax = 0.0
    ymin = 0.0
    ymax = 0.0
    for pts_it in [pts,pts1,pts2,pts3]:
        xmin_temp = min(pts_it, key=lambda item: item[0])[0] - 0.1
        xmax_temp = max(pts_it, key=lambda item: item[0])[0] + 0.1
        ymin_temp = min(pts_it, key=lambda item: item[1])[1] - 1
        ymax_temp = max(pts_it, key=lambda item: item[1])[1] + 1
        
        if xmin_temp < xmin:
            xmin = xmin_temp
        if xmax_temp > xmax:
            xmax = xmax_temp
        if ymin_temp < ymin:
            ymin = ymin_temp
        if ymax_temp > ymax:
               ymax = ymax_temp
    
    plt.xlabel('X [in]')
    plt.ylabel('Loads [var]')
    plt.title('Beam Socket Loads')
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis([xmin, xmax,ymin, ymax])
    plt.show()  
        
def pin_socket_stress(df,D_pin,Di_socket,t_socket):
    A_pin, I_pin = solid_cylinder_properties(D_pin)
    A_socket, I_socket = hollow_cylinder_properties(Di_socket, t_socket)
    
    c_pin = 0.5*D_pin
    c_socket = 0.5*(Di_socket+(2*t_socket))
    
    df['A_pin'] = A_pin
    df['I_pin'] = I_pin
    df['A_socket'] = A_socket
    df['I_socket'] = I_socket
    
    # Stress Calculations [ksi]
    df['f_s_pin'] = (df['Vx_pin'].abs()) / df['A_pin'] / 1000
    df['f_b_pin'] = df['Mx_pin'].abs()*c_pin / df['I_pin'] / 1000
    df['f_s_socket'] = (df['Vx_socket'].abs()) / df['A_socket'] / 1000
    df['f_b_socket'] = df['Mx_socket'].abs()*c_socket / df['I_socket'] / 1000

    return df

def pin_socket_margins(df,FF,Fs_pin, Fb_pin,Fs_socket, Fb_socket):
    # Allowables need to be in KSI
    df['Fs_pin'] = Fs_pin
    df['Fb_pin'] = Fb_pin
    df['Fs_socket'] = Fs_socket
    df['Fb_socket'] = Fb_socket
    
    df['MS_s_pin'] = (df['Fs_pin'].abs() / (df['f_s_pin']*FF)) - 1
    df['MS_b_pin'] = (df['Fb_pin'].abs() / (df['f_b_pin']*FF)) - 1
    df['MS_s_socket'] = (df['Fs_socket'].abs() / (df['f_s_socket']*FF)) - 1
    df['MS_b_socket'] = (df['Fb_socket'].abs() / (df['f_b_socket']*FF)) - 1
    
    df['Rs_pin'] = (df['f_s_pin']*FF) / df['Fs_pin'].abs()
    df['Rb_pin'] = (df['f_b_pin']*FF) / df['Fb_pin'].abs()
    df['MS_sb_pin'] = (1/(((df['Rs_pin']**2)+(df['Rb_pin']**2))**0.5)) - 1
    
    df['Rs_socket'] = (df['f_s_socket']*FF) / df['Fs_socket'].abs()
    df['Rb_socket'] = (df['f_b_socket']*FF) / df['Fb_socket'].abs()
    df['MS_sb_socket'] = (1/(((df['Rs_socket']**2)+(df['Rb_socket']**2))**0.5)) - 1

    return df

def convert_to_numeric(x):
    """
    Script to replace pd.to_numeric(df, errors ='ignore') 
         - With pandas 2.2: “ignore” is deprecated. Catch exceptions explicitly instead.
    """
    try:
        return pd.to_numeric(x)
    except:
        return x

def print_df_to_csv(df,filename,oDir):
    print('Printing data to CSV...')
    timestr = time.strftime("%m%d%Y")
    csv_path = oDir + filename + timestr + '.csv'
    df.to_csv(csv_path)
    pass

def create_or_increment_folder(oDir,folder_name):
    """Creates a folder if it doesn't exist. If it exists, adds an integer suffix."""
    
    if not os.path.exists(oDir + folder_name):
        os.makedirs(oDir + folder_name)
        oDir = oDir + folder_name + "//"
    else:
        itr = 1
        new_oDir = oDir
        while os.path.exists(new_oDir):
            new_oDir = oDir + folder_name + "_" + str(itr) + "//"
            itr += 1
        oDir = new_oDir
        os.makedirs(oDir)
        
    return oDir
    # Example usage
    #create_or_increment_folder("my_folder")

def output_xs_plot(df,pts,pts1,pts2,pts3,c,title,oDir):
    plt_lst = []
    
    # Plotting the generated points for visualization
    x, y = zip(*pts)
    x1, y1 = zip(*pts1)
    x2, y2 = zip(*pts2)
    x3, y3 = zip(*pts3)
    x4, y4 = zip(*c)
    
    plt.figure()
    plt.plot(x, y, 'b', label='Vx_pin')
    plt.plot(x1, y1, 'r', label='Mx_pin')
    plt.plot(x2, y2, 'b--', label='Vx_socket')
    plt.plot(x3, y3, 'r--', label='Mx_socket')
    plt.plot(x4, y4, 'bo', label='c_load_reversal')
    
    xmin = 0.0
    xmax = 0.0
    ymin = 0.0
    ymax = 0.0
    for pts_it in [pts,pts1,pts2,pts3]:
        xmin_temp = min(pts_it, key=lambda item: item[0])[0] - 0.1
        xmax_temp = max(pts_it, key=lambda item: item[0])[0] + 0.1
        ymin_temp = min(pts_it, key=lambda item: item[1])[1] - 1000
        ymax_temp = max(pts_it, key=lambda item: item[1])[1] + 1000
        
        if xmin_temp < xmin:
            xmin = xmin_temp
        if xmax_temp > xmax:
            xmax = xmax_temp
        if ymin_temp < ymin:
            ymin = ymin_temp
        if ymax_temp > ymax:
               ymax = ymax_temp
    
    plt.xlabel('X [in]')
    plt.ylabel('Loads [var]')
    plt.title('Beam Socket Loads - ' + title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis([xmin, xmax,ymin, ymax])

    # Save the plot to the specified directory
    plt.savefig(oDir + 'xs_' + title +   '.jpeg')
    plt.close()

#------------MARGIN OF SAFETY SUMMARY BY MIN OVERALL (s,t,st,br1,br2)---------#
def get_ms_summary_df_fm(df):
    print('Generating margin of safety summary by failure mode...')

    # MARGIN OF SAFETY SUMMARY BY FAILURE MODE
    MS_min_s_pin = []
    MS_min_b_pin = []
    MS_min_s_socket = []
    MS_min_b_socket = []
    MS_min_pin_sb = []
    MS_min_socket_sb = []
    
    MS_min_s_pin.append(df.nsmallest(1,'MS_s_pin', keep='first'))
    MS_min_b_pin.append(df.nsmallest(1,'MS_b_pin', keep='first'))
    MS_min_s_socket.append(df.nsmallest(1,'MS_s_socket', keep='first'))
    MS_min_b_socket.append(df.nsmallest(1,'MS_b_socket', keep='first'))
    MS_min_pin_sb.append(df.nsmallest(1,'MS_sb_pin', keep='first'))
    MS_min_socket_sb.append(df.nsmallest(1,'MS_sb_socket', keep='first'))
    
    lists = [MS_min_s_pin[0].iloc[0], MS_min_b_pin[0].iloc[0], MS_min_s_socket[0].iloc[0],MS_min_b_socket[0].iloc[0],MS_min_pin_sb[0].iloc[0],MS_min_socket_sb[0].iloc[0]]
    MS_fm_df = pd.concat([pd.Series(x) for x in lists], axis=1)
    MS_fm_df = MS_fm_df.T
    #MS_fm_df.drop(['index','F_X [lb]', 'F_Y [lb]','F_Z [lb]'], axis=1, inplace=True)
    
    # FORMATTING DATAFRAME FOR OUTPUT
    fm_list = ['Pin_Shear','Pin_Bending','Socket_Shear','Socket_Bending','Pin_S+B','Socket_S+B']
    MS_fm_df['Failure_Mode'] = fm_list
    MS_fm_df = MS_fm_df.set_index('Failure_Mode')
    # MS_fm_df = MS_fm_df[['Joint_ID','EID','Fast_Type','Fast_Size','SUBCASE',
    #                      'p_s_ult [lb] ','P_ss_final [lb]','MS_s',
    #                      'p_t_ult [lb] ','P_t_final [lb]','MS_t',
    #                      'Rs','Rt','Shear_Exp','Ten_Exp','MS_s&t',
    #                      'p_br1_ult [lb]','Pbru1_final [lb]','MS_bru1','t_req1_ult [in]',
    #                      'p_br1_lim [lb]','Pbry1_final [lb]','MS_bry1','t_req1_lim [in]',
    #                      'p_br2_ult [lb]','Pbru2_final [lb]','MS_bru2','t_req2_ult [in]',
    #                      'p_br2_lim [lb]','Pbry2_final [lb]','MS_bry2','t_req2_lim [in]',
    #                      'p_t_col_ult [lb]','P_t_col_final [lb]','MS_col_t','h5_file']]
    
    # CREATING A SEPARATE DATAFRAME TRUNCATED FOR VIEWING MARGINS
    MS_fm_reduced_df = MS_fm_df.copy(deep=True)
    MS_fm_reduced_df = MS_fm_reduced_df[['ID','SC','MS_s_pin','MS_b_pin','MS_s_socket','MS_b_socket','Rs_pin','Rb_pin','MS_sb_pin','Rs_socket','Rb_socket','MS_sb_socket']]   
    
    return MS_fm_df, MS_fm_reduced_df

def program_complete(string):
    #--------------------PROGRAM COMPLETED COMMUNICATION--------------------------#
    # PROGRAM COMPLETED
    print("Program Complete...")
    end_time = round((time.time()-start_time),2)
    print("--- %s seconds ---" % end_time)
    root = tk.Tk()
    tk.messagebox.showinfo(string, "Program Completed: %s seconds" % end_time)
    root.destroy()

#-----------------------------------------------------------------------------#
#---------------------------MAIN PROGRAM--------------------------------------#
#-----------------------------------------------------------------------------#
# Get user csv data
input_filename, oDir, user_input_df = get_user_input()

# Main loop for beam socket analysis - iterates over input rows
dic_temp = {}

# Generate images figures of XS
dir_img = create_or_increment_folder(oDir,'o_beam_socket_vm_images')

for i in range(user_input_df.shape[0]):
    df_final, c_o, plt_Vx1_lst, plt_Mx1_lst, plt_Vx2_lst, plt_Mx2_lst = NASA_beam_socket(user_input_df['ID'].iloc[i],user_input_df['SC'].iloc[i],user_input_df['Fo [lb]'].iloc[i],user_input_df['Mo [in-lb]'].iloc[i],user_input_df['L [in]'].iloc[i],user_input_df['a [in]'].iloc[i],user_input_df['b [in]'].iloc[i],user_input_df['n'].iloc[i])
    #plot_xs_pts(x_Vx1_lst, x_Mx1_lst, x_Vx2_lst, x_Mx2_lst, [(c_o,0)])
    df_sig = pin_socket_stress(df_final,user_input_df['D_pin [in]'].iloc[i],user_input_df['Di_socket [in]'].iloc[i],user_input_df['t_socket [in]'].iloc[i])
    df_ms = pin_socket_margins(df_sig,user_input_df['FF'].iloc[i],user_input_df['Fs_pin [ksi]'].iloc[i],user_input_df['Fb_pin [ksi]'].iloc[i], user_input_df['Fs_socket [ksi]'].iloc[i],user_input_df['Fb_socket [ksi]'].iloc[i])
    
    # Store XS data into dictionaries
    dict_key = str(user_input_df['ID'].iloc[i]) + "_" + str(user_input_df['SC'].iloc[i])
    dic_temp[dict_key] = df_ms # final dataframe with MS

    # Create VM image
    title = dict_key
    output_xs_plot(user_input_df,plt_Vx1_lst, plt_Mx1_lst, plt_Vx2_lst, plt_Mx2_lst,[(c_o,0)],title,dir_img)
    
bs_data_comb_df = pd.concat(dic_temp)
bs_data_comb_df = bs_data_comb_df.apply(convert_to_numeric)

# Get min ms dataframe
MS_fm_df, MS_fm_reduced_df = get_ms_summary_df_fm(bs_data_comb_df)

# Print dataframe to csv (all analysis data)
print_df_to_csv(bs_data_comb_df,'beam_socket_analysis_o_',oDir)
# Print dataframe to csv (ms data)
print_df_to_csv(MS_fm_df,'beam_socket_ms_summary_o_',oDir)
print_df_to_csv(MS_fm_reduced_df,'beam_socket_ms_red_summary_o_',oDir)
# Program completed
program_complete('Beam Socket Analysis Completed:')
