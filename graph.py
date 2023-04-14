import matplotlib.pyplot as plt


marker_list = ['o', 's', '*', 'x', 'D', '+']
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plot_r2(reg_acces, n_acces, mn_acces, p_acces, mp_acces, list_var):
    plt.figure(figsize=(15, 7))
    
    plt.plot(list_var, reg_acces, 'o-', color = color_list[0], label = 'Ideal')
    plt.plot(list_var, mp_acces, 's-', color = color_list[3], label = 'Multi modal w/ ED')
    plt.plot(list_var, mn_acces, 's--', color = color_list[3], label = 'Single modal w/ ED')
    plt.plot(list_var, p_acces, 'H-', color = color_list[4], label = 'Multi modal w/o ED')
    plt.plot(list_var, n_acces, 'H--', color = color_list[4], label = 'Single modal w/o ED')
    
    plt.xticks(list_var)
    plt.xlabel('Probability of Modal Imputation', fontsize=13)
    plt.ylabel('$R^2 Score$', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    plt.show()


def plot_r2_comb(n_acces, mn_acces, p_acces, mp_acces, list_var):
    plt.figure(figsize=(15, 7))
    plt.plot(list_var, n_acces, '*-', label = 'Single modal w/o ED')
    plt.plot(list_var, mn_acces, 's-', label = 'Single modal w/ ED')
    plt.plot(list_var, p_acces, 'h-', label = 'Multi modal w/o ED')
    plt.plot(list_var, mp_acces, 'H-', label = 'Multi modal w/ ED')
    
    plt.xticks(list_var, fontsize=12)
    plt.xlabel('Combination of Modal Imputation', fontsize=13)
    plt.ylabel('$R^2 Score$', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    plt.show()