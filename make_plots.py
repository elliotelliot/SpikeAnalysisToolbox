import numpy as np
import matplotlib.pyplot as plt



def make_plots_rates_adjusted(rates_adjusted, title_suplot):
    
    LAYERS = 4 
    STIMULI = 16
    for stimulus in range(STIMULI): 
        for layer in range(LAYERS):
            index = (stimulus * LAYERS)+ layer+1
            plt.suptitle(title_suplot, fontsize = 22, fontweight = "bold")
            plt.subplot(STIMULI, LAYERS, index)
            current_title = "Layer ", str(layer+1), "Stimulus " , str(stimulus+1)
            plt.title(current_title, fontsize = 18, fontweight = 'bold')
            _= plt.plot(rates_adjusted[layer][stimulus])
#    info = title_suplot +'Compare_Activation_Layer_Stimulus'
#    plt.savefig('images/' + info + '.png')



def make_plots_rates_filtered(filtered_rates_all, title_suplot):
    LAYERS = 4 
    STIMULI = 16
    for stimulus in range(STIMULI): 
        for layer in range(LAYERS):
            index = (stimulus * LAYERS)+ layer+1
            plt.suptitle(title_suplot, fontsize = 22, fontweight = "bold")
            plt.subplot(STIMULI, LAYERS, index)
            current_title = "Layer ", str(layer+1), "Stimulus " , str(stimulus+1)
            plt.title(current_title, fontsize = 12, fontweight = 'bold')
            _= plt.hist(filtered_rates_all[layer][stimulus], bins = 100)
            


def plot_single_cell(information_all_layers, num_categories):
    plt.figure(figsize=(20., 25.0))
    num_layers = len(information_all_layers)

    for layer in range(num_layers):
        current_information = information_all_layers[layer]    
        for cat_index in range(num_categories):
            plt.suptitle("Number of cells with single cell information", fontsize = 30, fontweight = "bold")
            index = (layer * num_categories)+(cat_index + 1)
            plt.subplot(num_layers, num_categories, index)
            plt.ylim(0.0, 1.1)

            if cat_index == 0: 
                plt.title("Position Layer " + str(layer+1) , fontsize = 22 )
                plt.text(0 , 0.5, "Red = Left Pos.")
                plt.text(0 , 0.4, "Blue = Right Pos.")
            elif cat_index == 1:
                plt.title("Border Layer " + str(layer+1), fontsize = 22)
                plt.text(0 , 0.5, "Red = Left Bor.")
                plt.text(0 , 0.4, "Blue = Right Bor.")
            elif cat_index == 2:
                plt.title("Color Layer " + str(layer+1), fontsize = 22)
                plt.text(0 , 0.5, "Red = Light BG")
                plt.text(0 , 0.4, "Blue = Dark BG.")
            elif cat_index == 3:
                plt.title("Shape Layer " + str(layer+1), fontsize = 22)
                plt.text(0 , 0.5, "Red = Dim.")
                plt.text(0 , 0.4, "Blue = Circl.")

            plt.plot(np.sort(current_information[cat_index][0]), color = 'r', linewidth = 5.0, alpha=0.5)
            plt.plot(np.sort(current_information[cat_index][1]), color = 'b', linewidth = 5.0, alpha=0.5)
         

