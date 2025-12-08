import tkinter as tk
from numpy import random as rn
import numpy as np
import csv

# Simulation parameters
TIME_SLOTS = 100000
PACKET_PROBABILITY = 0.005
PACKET_LENGTH = 4
NUM_NODES = 3
TIME_STEP_MS = 200

# Protocol constants
DIFS_DURATION = 3
SIFS_DURATION = 1
RTS_CTS_ACK_OVERHEAD = 5 # SIFS + CTS + SIFS + ACK

# Node activity states
STATE_IDLE = 0
STATE_SENDING_DATA = 1
STATE_WAITING_NAV = 2
STATE_WAITING_DIFS = 3
STATE_SENDING_RTS = 4
STATE_BACKOFF = 5

# Receiver states
RECEIVER_IDLE = 0
RECEIVER_SENDING_CTS = 1
RECEIVER_SENDING_ACK = 2
RECEIVER_COLLISION = 3


PACKETS_SENT = [0] * NUM_NODES
TOT_PACKETS = [0] * NUM_NODES
COLLISIONS = [0] * NUM_NODES

rn.seed(0)

list_packets_sent = []
list_tot_packets = []
list_collisions = []

class RTSPacket:
    def __init__(self, sender_id, data_length):
        self.sender_id = sender_id
        self.total_duration = data_length + RTS_CTS_ACK_OVERHEAD


class CTSPacket:
    def __init__(self, destination_id):
        self.destination_id = destination_id


class DataPacket:
    def __init__(self, sender_id, is_last=False):
        self.sender_id = sender_id
        self.is_last = is_last


class ACKPacket:
    def __init__(self, destination_id):
        self.destination_id = destination_id


class QLearningBackoff:
    """Q-Learning agent for adaptive backoff duration"""
    def __init__(self, min_backoff=1, max_backoff=10):
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff
        
        # State: number of recent collisions (0, 1, 2, 3+)
        self.num_states = 4
        # Actions: backoff durations from min to max
        self.num_actions = max_backoff - min_backoff + 1
        
        # Q-table: [state, action] -> expected reward
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2  # exploration rate
        
        # State tracking
        self.recent_collisions = 0
        self.last_state = 0
        self.last_action = 0
        
    def get_state(self):
        """Convert collision count to state index"""
        return min(self.recent_collisions, 3)
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if rn.random() < self.epsilon:
            # Explore: random action
            action = rn.randint(0, self.num_actions)
        else:
            # Exploit: best known action
            action = np.argmax(self.q_table[state])
        return action
    
    def action_to_backoff(self, action):
        """Convert action index to actual backoff duration"""
        return self.min_backoff + action
    
    def update(self, state, action, reward, next_state):
        """Q-learning update rule"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
    
    def get_backoff_duration(self):
        """Get backoff duration using Q-learning"""
        state = self.get_state()
        action = self.choose_action(state)
        
        self.last_state = state
        self.last_action = action
        
        return self.action_to_backoff(action)
    
    def report_success(self):
        """Report successful transmission"""
        next_state = 0  # Reset collision count after success
        reward = 10  # Large positive reward for success
        
        self.update(self.last_state, self.last_action, reward, next_state)
        self.recent_collisions = 0
    
    def report_collision(self):
        """Report collision"""
        self.recent_collisions += 1
        next_state = self.get_state()
        reward = -5  # Negative reward for collision
        
        self.update(self.last_state, self.last_action, reward, next_state)


class TransmitterNode:
    def __init__(self, node_id, packet_prob, packet_length):
        self.node_id = node_id
        self.packet_prob = packet_prob
        self.packet_length = packet_length
        
        # State variables
        self.data_queue = 0
        self.nav_timer = 0
        self.difs_timer = 0
        self.backoff_timer = 0
        self.data_remaining = 0
        
        # Status flags
        self.waiting_for_cts = False
        self.has_cts = False
        self.ready_to_send = False
        
        # Q-Learning agent replaces exponential backoff
        self.rl_agent = QLearningBackoff(min_backoff=1, max_backoff=10)
        
    def generate_new_data(self):
        if rn.random() < self.packet_prob:
            self.data_queue += 1
            TOT_PACKETS[int(self.node_id)-1] += 1
    
    def set_rl_backoff(self):
        """Use Q-learning to determine backoff duration"""
        self.backoff_timer = self.rl_agent.get_backoff_duration()
    
    def process_timestep(self, channel, t, activity_log):
        self.generate_new_data()
        current_channel = channel[t]
        prev_channel = channel[t-1] if t > 0 else None
        
        # If NAV is initialized, I must wait
        if self.nav_timer > 0:
            self.nav_timer -= 1
            self.waiting_for_cts = False
            activity_log[t] = STATE_WAITING_NAV
            return
        
        # If I'm ready to send data (I received a CTS)
        if self.ready_to_send:
            self.ready_to_send = False
            return
        
        # If I detected a collision in previous slot
        if isinstance(prev_channel, RTSPacket) and prev_channel.sender_id == "COLLISION":
            self._handle_collision(t, activity_log)
            return
        
        # If I received RTS from another node, set NAV
        if isinstance(prev_channel, RTSPacket) and prev_channel.sender_id != self.node_id:
            self._handle_others_rts(prev_channel, t, activity_log)
            return
        
        # If I have received a CTS
        if isinstance(current_channel, CTSPacket) and current_channel.destination_id == self.node_id:
            self._handle_cts_received(activity_log, t)
            return
        
        # Sending data
        if self.has_cts and self.data_remaining > 0:
            self._send_data(channel, t, activity_log)
            return
        
        # Finished sending data
        if self.has_cts and self.data_remaining == 0:
            self._finish_transmission()
            return
        
        # Waiting for CTS response
        if self.waiting_for_cts:
            self.waiting_for_cts = False
            return
        
        # DIFS countdown
        if self.difs_timer > 0:
            self._handle_difs(channel, t, activity_log)
            return
        
        # Backoff countdown
        if self.backoff_timer > 1:
            self.backoff_timer -= 1
            activity_log[t] = STATE_BACKOFF
            return
        
        # Backoff completed; try to send RTS
        if self.backoff_timer == 1:
            self.backoff_timer = 0
            activity_log[t] = STATE_BACKOFF
            if self.data_queue > 0:
                self._try_send_rts_after_backoff(channel, t, activity_log)
            return
        
        # If the channel is idle and I have data to send, start DIFS
        if current_channel is None and self.data_queue > 0 and self.difs_timer == 0:
            self.difs_timer = DIFS_DURATION
            activity_log[t] = STATE_WAITING_DIFS
            # Report success to RL agent on ACK
            if isinstance(prev_channel, ACKPacket):
                self.rl_agent.report_success()
    
    def _handle_collision(self, t, activity_log):
        self.difs_timer = 0
        self.waiting_for_cts = False
        COLLISIONS[int(self.node_id)-1] += 1
        # Report collision to RL agent
        self.rl_agent.report_collision()
        
        # Use RL-based backoff
        self.set_rl_backoff()
        
        # Start backoff immediately
        if self.backoff_timer > 0:
            activity_log[t] = STATE_BACKOFF
            #self.backoff_timer -= 1
        else:
            activity_log[t] = STATE_IDLE
    
    def _handle_others_rts(self, rts_packet, t, activity_log):
        self.nav_timer = rts_packet.total_duration
        self.difs_timer = 0
        self.backoff_timer = 0
        activity_log[t] = STATE_WAITING_NAV
        self.nav_timer -= 1
    
    def _handle_cts_received(self, activity_log, t):
        self.has_cts = True
        self.ready_to_send = True
        self.data_remaining = self.packet_length
    
    def _send_data(self, channel, t, activity_log):
        self.data_remaining -= 1
        is_last = (self.data_remaining == 0)
        channel[t] = DataPacket(self.node_id, is_last)
        activity_log[t] = STATE_SENDING_DATA
    
    def _finish_transmission(self):
        self.has_cts = False
        self.data_queue -= 1
        PACKETS_SENT[int(self.node_id)-1] += 1
    
    def _handle_difs(self, channel, t, activity_log):
        self.difs_timer -= 1
        activity_log[t] = STATE_WAITING_DIFS
        
        if self.difs_timer == 0:
            # DIFS complete, try to send RTS
            if channel[t] is None:
                # Channel free, send RTS
                channel[t] = RTSPacket(self.node_id, self.packet_length)
                self.waiting_for_cts = True
                activity_log[t] = STATE_SENDING_RTS
            elif isinstance(channel[t], RTSPacket):
                # Collision detected
                channel[t] = RTSPacket("COLLISION", self.packet_length)
                self.waiting_for_cts = True
                activity_log[t] = STATE_SENDING_RTS
            else:
                # Channel busy with other traffic, stay idle
                activity_log[t] = STATE_IDLE
    
    def _try_send_rts_after_backoff(self, channel, t, activity_log):
        self.backoff_timer = 0
        
        if channel[t] is None:
            # Channel free, send RTS
            channel[t] = RTSPacket(self.node_id, self.packet_length)
            self.waiting_for_cts = True
            activity_log[t] = STATE_SENDING_RTS
        elif isinstance(channel[t], RTSPacket):
            # Collision detected
            channel[t] = RTSPacket("COLLISION", self.packet_length)
            self.waiting_for_cts = True
            activity_log[t] = STATE_SENDING_RTS
        else:
            # Channel busy with other traffic, stay idle
            activity_log[t] = STATE_IDLE


class ReceiverNode:
    def __init__(self):
        self.destination_id = None
        self.expected_duration = 0
        self.should_send_cts = False
        self.should_send_ack = False
    
    def process_timestep(self, channel, time_idx, receiver_log):
        current_channel = channel[time_idx]
        
        # Handle RTS reception
        if isinstance(current_channel, RTSPacket):
            self.destination_id = current_channel.sender_id
            self.expected_duration = current_channel.total_duration
            
            if self.destination_id == "COLLISION":
                self.should_send_cts = False
                receiver_log[time_idx] = RECEIVER_COLLISION
            else:
                self.should_send_cts = True
        
        # Send CTS after SIFS
        elif self.should_send_cts:
            if time_idx + 1 < len(channel):
                channel[time_idx + 1] = CTSPacket(self.destination_id)
                receiver_log[time_idx + 1] = RECEIVER_SENDING_CTS
            self.should_send_cts = False
        
        # Handle data reception
        elif isinstance(current_channel, DataPacket):
            self.destination_id = current_channel.sender_id
            if current_channel.is_last:
                self.should_send_ack = True
        
        # Send ACK after SIFS
        elif self.should_send_ack:
            if time_idx + 1 < len(channel):
                channel[time_idx + 1] = ACKPacket(self.destination_id)
                receiver_log[time_idx + 1] = RECEIVER_SENDING_ACK
            self.should_send_ack = False


def run_simulation(num_slots, num_nodes, packet_prob, packet_length):
    channel = [None] * num_slots
    receiver_log = [RECEIVER_IDLE] * num_slots
    node_logs = [[STATE_IDLE] * num_slots for _ in range(num_nodes)]
    
    # Create nodes
    nodes = [
        TransmitterNode(str(i + 1), packet_prob, packet_length)
        for i in range(num_nodes)
    ]
    receiver = ReceiverNode()
    
    # Run simulation
    for t in range(num_slots):
        for idx, node in enumerate(nodes):
            node.process_timestep(channel, t, node_logs[idx])
        receiver.process_timestep(channel, t, receiver_log)
        if t % 1000 == 0 and t > 0:
            list_packets_sent.append(PACKETS_SENT.copy())
            list_tot_packets.append(TOT_PACKETS.copy())
            list_collisions.append(COLLISIONS.copy())
    return channel, node_logs, receiver_log, nodes


class CSMAGui:
    def __init__(self, root, channel, node_activities, receiver, nodes, step_ms=200):
        self.root = root
        self.channel = channel
        self.node_activities = node_activities
        self.receiver = receiver
        self.nodes = nodes
        self.time_slots = len(channel)
        self.num_nodes = len(node_activities)
        self.step_ms = step_ms

        self.cell_width = 15
        self.row_height = 22
        self.left_margin = 70

        self.colors = {0: "white", 1: "light grey", 2: "red", 3: "green", 4: "light blue", 
                      5: "light green", 6: "dark olive green", 7: "yellow", 8: "dark turquoise", 
                      9: "grey", 10: "dark red"}

        rows = 2 + self.num_nodes
        width = self.left_margin + self.cell_width * self.time_slots
        height = self.row_height * (rows + 1 + len(self.colors)) + 200

        self.canvas = tk.Canvas(root, width=width, height=height, bg="white")
        self.canvas.pack()

        self.channel_rects = [None] * self.time_slots
        self.node_rects = [[None] * self.time_slots for _ in range(self.num_nodes)]
        self.receiver_rects = [None] * self.time_slots

        self.current_t = 0

        self.draw_static_grid()
        self.create_rectangles()
        self.draw_legend()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Start animation", command=self.start_animation).pack()
        tk.Button(btn_frame, text="Show Q-tables", command=self.show_q_tables).pack()

    def draw_static_grid(self):
        for row in range(0, 2 + self.num_nodes):
            y0 = row * self.row_height
            y1 = y0 + self.row_height

            if row == 0: label = "Channel"
            elif row == 1 + self.num_nodes: label = "Receiver"
            else: label = "Node " + str(row)

            self.canvas.create_text(self.left_margin - 5, (y0 + y1) / 2, text=label, anchor="e")

            for t in range(0, self.time_slots, 5):
                x = self.left_margin + t * self.cell_width
                self.canvas.create_line(x, 0, x, self.row_height * (2 + self.num_nodes), dash=(2, 4))
                self.canvas.create_text(x + 2, self.row_height * (2 + self.num_nodes),
                                      text=str(t), anchor="n", font=("Arial", 7))

    def create_rectangles(self):
        for t in range(self.time_slots):
            y0 = 0 * self.row_height
            y1 = y0 + self.row_height
            x0 = self.left_margin + t * self.cell_width
            x1 = x0 + self.cell_width

            rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, fill="light grey", outline="black")
            self.channel_rects[t] = rect_id

            for row in range(self.num_nodes):
                y0 = (row + 1) * self.row_height
                y1 = y0 + self.row_height
                rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")
                self.node_rects[row][t] = rect_id

            y0 = (1 + self.num_nodes) * self.row_height
            y1 = y0 + self.row_height
            rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")
            self.receiver_rects[t] = rect_id

    def draw_legend(self):
        c = 0
        for row in range(3 + self.num_nodes, len(self.colors) + 3 + self.num_nodes):
            y0 = row * self.row_height
            y1 = y0 + self.row_height
            x0 = self.left_margin
            x1 = x0 + self.cell_width

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=self.colors[c], outline="black")
            
            labels = ["The node is waiting", "The channel is idle", "The channel is being used",
                     "Data packet", "Waiting for DIFS", "RTS packet", "CTS packet", "NAV",
                     "ACK", "RL Backoff", "Collision"]
            
            self.canvas.create_text(x1 + len(self.colors), (y0 + y1) / 2, 
                                  text=labels[c], anchor="w")
            c += 1

    def show_q_tables(self):
        """Display Q-tables for first 3 nodes"""
        window = tk.Toplevel(self.root)
        window.title("Q-Learning Tables")
        
        text = tk.Text(window, width=80, height=30, font=("Courier", 10))
        text.pack(padx=10, pady=10)
        
        for i in range(min(3, len(self.nodes))):
            node = self.nodes[i]
            text.insert(tk.END, f"\n{'='*60}\n")
            text.insert(tk.END, f"Node {node.node_id} Q-Table\n")
            text.insert(tk.END, f"{'='*60}\n\n")
            text.insert(tk.END, f"States: 0=No collisions, 1=1 collision, 2=2 collisions, 3=3+ collisions\n")
            text.insert(tk.END, f"Actions: Backoff duration (1-10 time slots)\n\n")
            
            q_table = node.rl_agent.q_table
            text.insert(tk.END, "      " + "".join([f"Act{j:2d} " for j in range(q_table.shape[1])]) + "\n")
            
            for state in range(q_table.shape[0]):
                text.insert(tk.END, f"St{state}: ")
                for action in range(q_table.shape[1]):
                    text.insert(tk.END, f"{q_table[state, action]:6.2f} ")
                text.insert(tk.END, "\n")
            
            text.insert(tk.END, f"\nBest actions per state:\n")
            for state in range(q_table.shape[0]):
                best_action = np.argmax(q_table[state])
                best_backoff = node.rl_agent.action_to_backoff(best_action)
                text.insert(tk.END, f"  State {state}: Backoff {best_backoff} slots\n")
        
        text.config(state=tk.DISABLED)

    def start_animation(self):
        for t in range(self.time_slots):
            self.canvas.itemconfig(self.channel_rects[t], fill="light grey")
            for row in range(self.num_nodes):
                self.canvas.itemconfig(self.node_rects[row][t], fill="white")
            self.canvas.itemconfig(self.receiver_rects[t], fill="white")
        self.current_t = 0
        self.animate_step()

    def animate_step(self):
        if self.current_t >= self.time_slots:
            return

        t = self.current_t

        channel_busy = self.channel[t]
        chan_color = "red" if channel_busy else "light grey"
        self.canvas.itemconfig(self.channel_rects[t], fill=chan_color)
        
        node_colors = {0: "white", 1: "green", 2: "yellow", 3: "light blue", 
                      4: "light green", 5: "grey"}
        for row in range(self.num_nodes):
            status = self.node_activities[row][t]
            color = node_colors[status]
            self.canvas.itemconfig(self.node_rects[row][t], fill=color)

        receiver_colors = {0: "white", 1: "dark olive green", 2: "dark turquoise", 3: "dark red"}
        status = self.receiver[t]
        color = receiver_colors[status]
        self.canvas.itemconfig(self.receiver_rects[t], fill=color)
        
        self.current_t += 1
        self.root.after(self.step_ms, self.animate_step)

if __name__ == "__main__":
    channel, node_activities, receiver, nodes = run_simulation(
        num_slots=TIME_SLOTS,
        num_nodes=NUM_NODES,
        packet_prob=PACKET_PROBABILITY,
        packet_length=PACKET_LENGTH
    )

    csv.writer(open("data/packets_sent_RL.csv", "w", newline="")).writerows(list_packets_sent)
    csv.writer(open("data/collisions_RL.csv", "w", newline="")).writerows(list_collisions)
    csv.writer(open("data/tot_packets_RL.csv", "w", newline="")).writerows(list_tot_packets)

    if False:
        root = tk.Tk()
        root.title("CSMA-CA with Q-Learning Backoff")
        app = CSMAGui(root, channel, node_activities, receiver, nodes, step_ms=TIME_STEP_MS)
        root.mainloop()