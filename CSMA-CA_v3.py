import tkinter as tk
from numpy import random as rn

# The parameters
time_slots = 50 # the number of time slots
p_packet = 0.1 # the probability of sending a packet
packet_length = 4 # the length of the packet
num_nodes = 3 # the number of nodes
time_step_ms = 200 # the delay between time slots in ms

counter_DIFS = 3

class TransmitterNode:
    def __init__(self, p_packet=0.3, channel=None, packet_length=5, name="0"):
        self.p_packet = p_packet
        self.channel = channel
        self.packet_length = packet_length
        self.name = name
        self.RTS = False

    def try_send_data(self, t, send_data, counter, node_activity):
        # if the channel is idle and we haven't sent the RTS and we want to send data
        if (not self.channel[t]) and (not self.RTS) and send_data:
            if counter > 1: # if we are already waiting for DIFS, decrease 1
                counter -= 1
            elif not counter: # if we just sensed the channel, wait for DIFS
                counter = counter_DIFS
                for j in range(t,t + counter):
                    if j < len(self.channel):
                        node_activity[j] = 2
            elif counter == 1: # when we have waited for DIFS, send RTS
                self.RTS = True
                self.channel[t] = 1
                node_activity[t] = 3
        
        # if the channel is idle, we have sent the RTS and we want to send data
        elif (not self.channel[t]) and self.RTS and send_data:
            if counter == 0: # send the packet
                for j in range(t, t + self.packet_length): 
                    if j < len(self.channel):
                        self.channel[j] = 1
                        node_activity[j] = 1
                send_data -= 1 # remove from the queue the packet we just sent
                self.RTS = False
            elif counter == 1: # wait 1
                counter = 0
        else:
            counter = 0
        return send_data, counter


def run_simulation(timestamps, num_nodes, p_packet, packet_length):
    channel = [0] * timestamps # the common channel
    node_activities = [[0] * timestamps for _ in range(num_nodes)] # the list of the schedules of all the nodes

    nodes = [TransmitterNode(p_packet=p_packet, channel=channel, packet_length=packet_length, name=str(i + 1))
        for i in range(num_nodes)] # list of all the nodes
    send_data = [0] * num_nodes
    counters = [0] * num_nodes
    for t in range(timestamps):
        print("counters", counters)
        for idx, node in enumerate(nodes):
            send_data[idx] += int(rn.choice([0, 1], p=[1 - p_packet, p_packet])) # 1 if we want to send data
            if send_data[idx]:
               print(idx, t)
            send_data[idx], counters[idx] = node.try_send_data(t, send_data[idx], counters[idx], node_activities[idx]) # runs each node one after the other

    return channel, node_activities


class CSMAGui:
    def __init__(self, root, channel, node_activities, step_ms=200):
        self.root = root
        self.channel = channel
        self.node_activities = node_activities
        self.time_slots = len(channel)
        self.num_nodes = len(node_activities)
        self.step_ms = step_ms

        self.cell_width = 15
        self.row_height = 22
        self.left_margin = 70

        rows = 2 + self.num_nodes
        width = self.left_margin + self.cell_width * self.time_slots
        height = self.row_height * (rows + 6) + 10

        self.canvas = tk.Canvas(root, width=width, height=height, bg="white")
        self.canvas.pack()

        # initializes all the rectangles to None
        self.channel_rects = [None] * self.time_slots
        self.node_rects = [[None] * self.time_slots for _ in range(self.num_nodes)]
        self.receiver_rects = [None] * self.time_slots

        self.current_t = 0

        self.draw_static_grid() # creates the column with the labels
        self.create_rectangles() # creates the rectangles to be colored
        self.draw_legend() # draws the legend

        # button to restart animation
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Start animation", command=self.start_animation).pack()

    def draw_static_grid(self):
        for row in range(0, 2 + self.num_nodes):
            y0 = row * self.row_height
            y1 = y0 + self.row_height

            if row == 0: label = "Channel"
            elif row == 1 + self.num_nodes: label = "Receiver"
            else:        label = "Node " + str(row)

            # writes the labels on the left
            self.canvas.create_text(self.left_margin - 5, (y0 + y1) / 2, text=label, anchor="e")

            # draw small ticks every 5 slots
            for t in range(0, self.time_slots, 5):
                x = self.left_margin + t * self.cell_width
                self.canvas.create_line(x, 0, x, self.row_height * (2 + self.num_nodes), dash=(2, 4))
                self.canvas.create_text(x + 2, self.row_height * (2 + self.num_nodes),
                                        text=str(t), anchor="n", font=("Arial", 7))

    def create_rectangles(self):
        # initially all idle colors (grey for channel, white for nodes)
        for t in range(self.time_slots):
            # channel row
            y0 = 0 * self.row_height
            y1 = y0 + self.row_height
            x0 = self.left_margin + t * self.cell_width
            x1 = x0 + self.cell_width

            rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1,
                fill="light grey",
                outline="black")
            
            self.channel_rects[t] = rect_id

            # node rows
            for row in range(self.num_nodes):
                y0 = (row + 1) * self.row_height
                y1 = y0 + self.row_height
                rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill="white",
                    outline="black"
                )
                self.node_rects[row][t] = rect_id

            # receiver row
            y0 = (1 + self.num_nodes) * self.row_height
            y1 = y0 + self.row_height
            x0 = self.left_margin + t * self.cell_width
            x1 = x0 + self.cell_width

            rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1,
                fill="white",
                outline="black")
            
            self.receiver_rects[t] = rect_id


    def draw_legend(self):
        c = 0
        colors = {0: "white", 1: "grey", 2: "red", 3: "green", 4: "light blue", 5: "light green"}
        for row in range(3 + self.num_nodes, 5 + 3 + self.num_nodes):
            y0 = row * self.row_height
            y1 = y0 + self.row_height
            x0 = self.left_margin
            x1 = x0 + self.cell_width

            rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1,
                fill=colors[c],
                outline="black")
            
            if c == 0: label = "The node is waiting"
            elif c == 1: label = "The channel is idle"
            elif c == 2: label = "The channel is being used"
            elif c == 3: label = "Data packet"
            elif c == 4: label = "Waiting for DIFS"
            elif c == 5: label = "RTS packet"

            # writes the labels on the left
            self.canvas.create_text(x1 + 5, (y0 + y1) / 2, text=label, anchor="w")
            c += 1


    def start_animation(self):
        # reset all colors
        for t in range(self.time_slots):
            self.canvas.itemconfig(self.channel_rects[t], fill="light grey")
            for row in range(self.num_nodes):
                self.canvas.itemconfig(self.node_rects[row][t], fill="white")

        self.current_t = 0
        self.animate_step()

    def animate_step(self):
        if self.current_t >= self.time_slots:
            return

        t = self.current_t

        # Channel color at time t
        channel_busy = self.channel[t]
        chan_color = "red" if channel_busy else "light grey"
        self.canvas.itemconfig(self.channel_rects[t], fill=chan_color)

        colors = {0: "white", 1: "green", 2: "light blue", 3: "light green"}
        # Node colors at time t
        for row in range(self.num_nodes):
            status = self.node_activities[row][t]
            color = colors[status]
            self.canvas.itemconfig(self.node_rects[row][t], fill=color)

        self.current_t += 1
        self.root.after(self.step_ms, self.animate_step)


if __name__ == "__main__":
    # Run the simulation
    channel, node_activities = run_simulation(
        timestamps=time_slots,
        num_nodes=num_nodes,
        p_packet=p_packet,
        packet_length=packet_length
    )

    # Build GUI with animation
    root = tk.Tk()
    root.title("MAC Protocol Visualization (Animated)")
    app = CSMAGui(root, channel, node_activities, step_ms=time_step_ms)
    root.mainloop()
