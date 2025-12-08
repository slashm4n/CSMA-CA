import tkinter as tk
from numpy import random as rn

# The parameters
time_slots = 100 # the number of time slots
p_packet = 0.3 # the probability of sending a packet
packet_length = 4 # the length of the packet
num_nodes = 5 # the number of nodes
time_step_ms = 200 # the delay between time slots in ms


class RTSpacket:
    def __init__(self, parent_node, lenght_data):
        self.parent_node = parent_node
        self.length_data = lenght_data + 5 # + 5 because it's SIFS + CTS + SIFS + ACK


class CTSpacket:
    def __init__(self, destination_node):
        self.destination_node = destination_node


class Datapacket:
    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.end_packet = False # if it's the last data packet, it must be set to True 


class ACKpacket:
    def __init__(self):
        pass

class TransmitterNode:
    def __init__(self, p_packet=0.3, packet_length=5, name="0"):
        self.p_packet = p_packet
        self.packet_length = packet_length
        self.name = name
        self.RTS = False
        self.NAV = 0
        self.DIFS = 0
        self.CTS = False
        self.counter_data = packet_length
        self.waiting_for_CTS = False
        self.ready_to_send_data = False
        self.data_to_send = 0
        self.contention_window = 7
        self.exponential_backoff = 0

    def try_send_data(self, channel, t, node_activity):
        channel_status = channel[t]
        self.data_to_send += int(rn.choice([0, 1], p=[1 - p_packet, p_packet])) # 1 if we want to send data
        # If NAV is initialized I can't do anything else
        if self.NAV:
            self.NAV -= 1
        # If I'm ready to send data
        elif self.ready_to_send_data == True:
            self.ready_to_send_data = False
        # If there's a collision
        elif (type(channel[t-1]) == RTSpacket and channel[t-1].parent_node == "collision"):
                self.DIFS = 0
                self.exponential_backoff = rn.randint(4,self.contention_window)
                for i in range(t, t + self.exponential_backoff):
                    if i < len(channel):
                        node_activity[i] = 5 # the node activity is waiting for the contention window
        # If someone else sent an RTS, I wait for that amount of time
        elif type(channel[t-1]) == RTSpacket and channel[t-1].parent_node != self.name:
                self.NAV = channel[t-1].length_data
                self.DIFS = 0
                self.exponential_backoff = 0
                for i in range(t, t + self.NAV):
                    if i < len(channel):
                        node_activity[i] = 2 # the node activity is waiting for the NAV
                self.NAV -= 1
        elif self.exponential_backoff > 1:
            self.exponential_backoff -= 1
        # If I received a CTS for me
        elif type(channel_status) == CTSpacket and channel_status.destination_node == self.name:
            self.CTS = True
            self.ready_to_send_data = True
        # If I'm waiting to send an RTS
        elif self.DIFS > 0:
            self.DIFS -= 1
            node_activity[t] = 3
            # If I have waited for DIFS, send an RTS
            if self.DIFS == 0 and channel[t] == None:
                self.waiting_for_CTS = True
                RTS = RTSpacket(self.name, self.packet_length)
                channel[t] = RTS
                node_activity[t] = 4
            elif self.DIFS == 0 and type(channel[t]) == RTSpacket:
                node_activity[t] = 4
                self.waiting_for_CTS = True
                RTS = RTSpacket("collision", self.packet_length)
                channel[t] = RTS
            elif self.DIFS == 0 and channel[t] != None:
                node_activity[t] = 0
        # If I'm waiting to send an RTS after an exponential backoff
        elif self.exponential_backoff == 1:
            self.exponential_backoff = 0
            if channel[t] == None:
                self.waiting_for_CTS = True
                RTS = RTSpacket(self.name, self.packet_length)
                channel[t] = RTS
                node_activity[t] = 4
            elif self.DIFS == 0 and type(channel[t]) == RTSpacket:
                node_activity[t] = 4
                self.waiting_for_CTS = True
                self.contention_window += 2
                RTS = RTSpacket("collision", self.packet_length)
                channel[t] = RTS
            else:
                node_activity[t] = 0
        # If I'm ready to send a packet:
        elif self.CTS:
            # If I still have to send data
            if self.counter_data > 0:
                self.counter_data -= 1
                data = Datapacket(self.name)
                data.end_packet = False
                if self.counter_data == 0:
                    data.end_packet = True
                channel[t] = data # send the data
                node_activity[t] = 1
            # If I have finished sending data
            elif self.counter_data == 0:
                self.CTS = False
                self.counter_data = packet_length
                self.data_to_send -= 1
        # If I'm waiting for a CTS
        elif self.waiting_for_CTS == True:
            self.waiting_for_CTS = False
        # If the channel is idle and I want to send data
        elif channel_status == None and self.data_to_send and not self.DIFS:
                self.DIFS = 3
                node_activity[t] = 3

            
class ReceiverNode:
    def __init__(self):
        self.counter_packet = 0
        self.send_CTS = False
        self.destination_node = None
        self.send_ACK = False
    
    def listen(self, channel, t, receiver):
        channel_status = channel[t]
        if type(channel_status) == RTSpacket:
            self.destination_node = channel_status.parent_node
            self.counter_packet = channel_status.length_data
            if self.destination_node == "collision":
                self.send_CTS = False
                receiver[t] = 3
            else:    
                self.send_CTS = True
            
        elif self.send_CTS == True:
            CTS = CTSpacket(self.destination_node)
            if t + 1 < len(channel):
                channel[t+1] = CTS
                self.send_CTS = False
                receiver[t+1] = 1
        elif type(channel_status) == Datapacket:
            self.destination_node = channel_status.parent_node
            if channel_status.end_packet == True:
                self.send_ACK = True
        elif self.send_ACK == True:
            ACK = ACKpacket()
            if t + 1 < len(channel):
                channel[t+1] = ACK
                self.send_ACK = False
                receiver[t+1] = 2


def run_simulation(timestamps, num_nodes, p_packet, packet_length):
    channel = [None] * timestamps # the common channel
    receiver = [0] * timestamps # the reciver's activity

    node_activities = [[0] * timestamps for _ in range(num_nodes)] # the list of the schedules of all the nodes

    nodes = [TransmitterNode(p_packet=p_packet, packet_length=packet_length, name=str(i + 1))
        for i in range(num_nodes)] # list of all the nodes
    receiver_node = ReceiverNode() # instantiate the receiver node
    for t in range(timestamps):
        for idx, node in enumerate(nodes):
            node.try_send_data(channel, t, node_activities[idx])
            receiver_node.listen(channel, t, receiver)
    print(channel)
    return channel, node_activities, receiver


class CSMAGui:
    def __init__(self, root, channel, node_activities, receiver, step_ms=200):
        self.root = root
        self.channel = channel
        self.node_activities = node_activities
        self.receiver = receiver
        self.time_slots = len(channel)
        self.num_nodes = len(node_activities)
        self.step_ms = step_ms

        self.cell_width = 15
        self.row_height = 22
        self.left_margin = 70

        self.colors = {0: "white", 1: "light grey", 2: "red", 3: "green", 4: "light blue", 5: "light green", 6: "dark olive green", 7: "yellow", 8: "dark turquoise", 9: "grey", 10: "dark red"}

        rows = 2 + self.num_nodes
        width = self.left_margin + self.cell_width * self.time_slots
        height = self.row_height * (rows + 1 + len(self.colors)) + 10

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
        for row in range(3 + self.num_nodes, len(self.colors) + 3 + self.num_nodes):
            y0 = row * self.row_height
            y1 = y0 + self.row_height
            x0 = self.left_margin
            x1 = x0 + self.cell_width

            rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1,
                fill=self.colors[c],
                outline="black")
            
            if c == 0: label = "The node is waiting"
            elif c == 1: label = "The channel is idle"
            elif c == 2: label = "The channel is being used"
            elif c == 3: label = "Data packet"
            elif c == 4: label = "Waiting for DIFS"
            elif c == 5: label = "RTS packet"
            elif c == 6: label = "CTS packet"
            elif c == 7: label = "NAV"
            elif c == 8: label = "ACK"
            elif c == 9: label = "Exponential backoff"
            elif c == 10: label = "Collision"

            # writes the labels on the left
            self.canvas.create_text(x1 + len(self.colors), (y0 + y1) / 2, text=label, anchor="w")
            c += 1


    def start_animation(self):
        # reset all colors
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

        # Channel color at time t
        channel_busy = self.channel[t]
        chan_color = "red" if channel_busy else "light grey"
        self.canvas.itemconfig(self.channel_rects[t], fill=chan_color)
        
        # node_activity = 0 nothing, 1 data, 2, NAV, 3, DIFS, 4, RTS
        colors = {0: "white", 1: "green", 2: "yellow", 3: "light blue", 4: "light green", 5: "grey"}
        # Node colors at time t
        for row in range(self.num_nodes):
            status = self.node_activities[row][t]
            color = colors[status]
            self.canvas.itemconfig(self.node_rects[row][t], fill=color)

        colors = {0: "white", 1: "dark olive green", 2: "dark turquoise", 3: "dark red"}
        status = self.receiver[t]
        color = colors[status]
        self.canvas.itemconfig(self.receiver_rects[t], fill=color)
        
        self.current_t += 1
        self.root.after(self.step_ms, self.animate_step)


if __name__ == "__main__":
    # Run the simulation
    channel, node_activities, receiver = run_simulation(
        timestamps=time_slots,
        num_nodes=num_nodes,
        p_packet=p_packet,
        packet_length=packet_length
    )

    # Build GUI with animation
    root = tk.Tk()
    root.title("MAC Protocol Visualization (Animated)")
    app = CSMAGui(root, channel, node_activities, receiver, step_ms=time_step_ms)
    root.mainloop()
