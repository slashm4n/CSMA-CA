import matplotlib.pyplot as plt
from numpy import cumsum as cs
from numpy import insert

n, p = 25, 0.01

def running_mean(x, N):
    # In case we have too many data points, to avoid fluctuations we use a running mean over N points
    cumsum = cs(insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def read_csv_data(filename):
    # Open the CSV files
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append([int(x) for x in line.strip().split(',')])

    return data

# Read data from CSV files
list_packets_sent_RL = read_csv_data("data/packets_sent_RL.csv")
list_collisions_RL = read_csv_data("data/collisions_RL.csv")
list_tot_packets_RL = read_csv_data("data/tot_packets_RL.csv")

list_packets_sent_classic = read_csv_data("data/packets_sent_classic.csv")
list_collisions_classic = read_csv_data("data/collisions_classic.csv")
list_tot_packets_classic = read_csv_data("data/tot_packets_classic.csv")

list_std_classic = read_csv_data("data/backoffs_classic_n_" + str(n) + "_p_" + str(p) + ".csv")
list_std_RL = read_csv_data("data/backoffs_RL_n_" + str(n) + "_p_" + str(p) + ".csv")

def plot_one_result(list_packets_sent_classic, list_collisions_classic, list_tot_packets_classic):
    # Plot only one result set
    fig, axs = plt.subplots(3, 1, dpi=100, figsize=(8, 8))
    fig.tight_layout()

    n_nodes = len(list_packets_sent_classic[0])
    tot_classic = [sum(list_packets_sent_classic[i])/n_nodes for i in range(len(list_packets_sent_classic))]
    tot_collisions_classic = [list_collisions_classic[i][0]/n_nodes for i in range(len(list_collisions_classic))]
    tot_packet_ratio_classic = [(sum(list_packets_sent_classic[i]))/(sum(list_tot_packets_classic[i])) for i in range(len(list_tot_packets_classic))]

    axs[0].plot([1000 * (i+1) for i in range(len(list_packets_sent_classic))], tot_classic, color="blue")
    axs[0].set(xlabel = "Time Slots", ylabel = "Packets Sent")
    axs[0].set_title("Avg Packets Sent Over Time - Q-Learning Backoff - ε = 0.01 - p = 0.005 - n = " + str(n_nodes))
    axs[0].grid()

    axs[1].plot([1000 * (i+1) for i in range(len(tot_packet_ratio_classic))], tot_packet_ratio_classic, color = "red")
    axs[1].set(xlabel = "Time Slots", ylabel = "Packet Ratio")
    axs[1].set_title("Avg Packet Ratio Over Time - Q-Learning Backoff - ε = 0.01 - p = 0.005 - n = " + str(n_nodes))
    axs[1].grid()

    axs[2].plot([1000 * (i+1) for i in range(len(tot_collisions_classic))], tot_collisions_classic, color = "green")
    axs[2].set(xlabel = "Time Slots", ylabel = "Collisions")
    axs[2].set_title("Avg Collisions Over Time - Q-Learning Backoff - ε = 0.01 - p = 0.005 - n = " + str(n_nodes))
    axs[2].grid()

    fig.tight_layout()
    plt.savefig("Paper/imgs/RL_" + str(n_nodes) + "_nodes_ε_0.01_p_0.005.png")
    plt.show()


def plot_total_results(list_packets_sent_RL, list_packets_sent_classic, list_collisions_RL, list_collisions_classic, list_tot_packets_RL, list_tot_packets_classic):
    # Plot both results for comparison
    fig, axs = plt.subplots(3, 1, dpi=100, figsize=(8, 8), constrained_layout=True)
    #fig.tight_layout()

    n_nodes = len(list_packets_sent_classic[0])

    tot_packets_RL = [sum(list_packets_sent_RL[i])/100000 for i in range(len(list_packets_sent_RL))]
    tot_packets_classic = [sum(list_packets_sent_classic[i])/100000 for i in range(len(list_packets_sent_classic))]
    
    tot_collisions_RL = [list_collisions_RL[i][0] for i in range(len(list_collisions_RL))]
    tot_collisions_classic = [list_collisions_classic[i][0] for i in range(len(list_collisions_classic))]
    
    tot_packet_ratio_classic = [(sum(list_packets_sent_classic[i]))/(sum(list_tot_packets_classic[i])) for i in range(len(list_tot_packets_classic))]
    tot_packet_ratio_RL = [(sum(list_packets_sent_RL[i]))/(sum(list_tot_packets_RL[i])) for i in range(len(list_tot_packets_RL))]

    axs[0].plot([1000 * (i+1) for i in range(len(list_packets_sent_RL))], tot_packets_RL, color = "blue", label = "Q-Learning Backoff")
    axs[0].plot([1000 * (i+1) for i in range(len(list_packets_sent_classic))], tot_packets_classic, color = "red", label = "Classic CSMA-CA")
    axs[0].set(xlabel = "Time Slots", ylabel = "Throughput")
    axs[0].set_title("Throughput Over Time - ε = 0.01 - p = 0.01 - n = " + str(n_nodes))
    axs[0].legend()
    axs[0].grid()

    axs[1].plot([1000 * (i+1) for i in range(len(tot_packet_ratio_RL))], tot_packet_ratio_RL, color = "blue", label = "Q-Learning Backoff")
    axs[1].plot([1000 * (i+1) for i in range(len(tot_packet_ratio_classic))], tot_packet_ratio_classic, color = "red", label = "Classic CSMA-CA")
    axs[1].set(xlabel = "Time Slots", ylabel = "Packets Ratio")
    axs[1].set_title("Packet Ratio Over Time - ε = 0.01 - p = 0.01 - n = " + str(n_nodes))
    axs[1].legend()
    axs[1].grid()

    axs[2].plot([1000 * (i+1) for i in range(len(tot_collisions_RL))], tot_collisions_RL, color = "blue", label = "Q-Learning Backoff")
    axs[2].plot([1000 * (i+1) for i in range(len(tot_collisions_classic))], tot_collisions_classic, color = "red", label = "Classic CSMA-CA")
    axs[2].set(xlabel = "Time Slots", ylabel = "Collisions")
    axs[2].set_title("Total Collisions Over Time - ε = 0.01 - p = 0.01 - n = " + str(n_nodes))
    axs[2].legend()
    axs[2].grid()

    plt.savefig("Paper/imgs/Comparison_" + str(n_nodes) + "_nodes_ε_0.01_p_0.01.png")
    plt.show()


def plot_results_per_node(RL_data, classic_data, title, node_index):
    # Plot results for a specific node
    data_node_i_RL = [RL_data[i][node_index] for i in range(len(list_packets_sent_RL))]
    data_node_i_classic = [classic_data[i][node_index] for i in range(len(list_packets_sent_classic))]
    
    
    plt.plot([1000 * (i+1) for i in range(len(RL_data))], data_node_i_RL)
    plt.plot([1000 * (i+1) for i in range(len(classic_data))], data_node_i_classic)
    plt.xlabel("Time Slots")
    plt.ylabel("Packets Sent by Node " + str(node_index))
    plt.title(title)
    plt.legend(["Q-Learning Backoff", "Classic CSMA-CA"])
    plt.grid()
    plt.show()


def plot_standard_deviations(RL_data, classic_data, title):
    import numpy as np
    # Plot standard deviations over time
    std_dev_RL = [np.std(RL_data[i]) for i in range(len(RL_data))]
    std_dev_classic = [np.std(classic_data[i]) for i in range(len(classic_data))]

    std_dev_RL = running_mean(std_dev_RL, 1000)
    std_dev_classic = running_mean(std_dev_classic, 1000)

    plt.plot([1000 * (i+1) for i in range(len(std_dev_RL))], std_dev_RL)
    plt.plot([1000 * (i+1) for i in range(len(std_dev_classic))], std_dev_classic)
    plt.xlabel("Time Slots")
    plt.ylabel("Standard Deviation of Packets Sent")
    plt.title(title)
    plt.legend(["Q-Learning Backoff", "Classic CSMA-CA"])
    plt.grid()
    plt.show()


#for i in range(3):
    #plot_results(list_packets_sent_RL, list_packets_sent_classic, "Packets Sent Over Time by Node " + str(i), i)
plot_total_results(list_packets_sent_RL, list_packets_sent_classic, list_collisions_RL, list_collisions_classic, list_tot_packets_RL, list_tot_packets_classic)

#plot_results_per_node(list_packets_sent_RL, list_packets_sent_classic, "Packets Sent Over Time", 0)
#plot_results_per_node(list_collisions_RL, list_collisions_classic, "Collisions Over Time", 0)

#plot_one_result(list_packets_sent_classic, list_collisions_classic, list_tot_packets_classic)
#plot_one_result(list_packets_sent_RL, list_collisions_RL, list_tot_packets_RL)

#plot_standard_deviations(list_std_RL, list_std_classic, "Standard Deviation of Packets Sent Over Time")