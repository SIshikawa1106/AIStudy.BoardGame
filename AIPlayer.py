import BoardGame as game
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import random
import numpy as np
import matplotlib.pyplot as plt
import os

class RandomActor:
    def __init__(self, board):
        self.board = board
        self.random_count = 0

    def action_priority(self, pieces):
        cmdList = []
        cnt = 0
        for piece in pieces:
            for cmd in piece.npos:
                cmdList.append([cmd[0], [cmd, piece]])
                val = (self.board.rows - cmd[0] + 1)
                cnt += val*val
        rand = random.random() * cnt

        cnt = 0
        for cmd in cmdList:
            val = self.board.rows - cmd[0] + 1
            cnt += val * val
            if cnt > rand:
                return cmd[1][1], cmd[1][0]

    def action_random(self, pieces):
        piece = random.choice(pieces)
        cmd = random.choice(piece.npos)
        return piece, cmd

    def random_action_func(self):
        self.random_count += 1
        pieces = self.board.get_enable_pieces()
        #piece, cmd = self.action_priority(pieces)
        piece, cmd = self.action_random(pieces)

        #print(("random act:idx={},cmd={}").format(piece.PieceID, cmd))
        return cnvrtMoveCmd2ActCmd(self.board, piece.PieceID, [cmd[0], cmd[1]])

def cnvrtActCmd2MoveCmd(board, act):
    pid = int(act / 8) + 1
    target = board.get_target(pid)
    index = act % 8
    cmd = []
    row = target[0]
    col = target[1]
    # print(("target = {}").format(target))
    if index == 0:
        cmd = [row, col - 1]
    elif index == 1:
        cmd = [row - 1, col - 1]
    elif index == 2:
        cmd = [row - 1, col]
    elif index == 3:
        cmd = [row - 1, col + 1]
    elif index == 4:
        cmd = [row, col + 1]
    elif index == 5:
        cmd = [row +1, col + 1]
    elif index == 6:
        cmd = [row +1, col]
    else:
        cmd = [row +1, col -1]

    #print(("act={},pid={},index={},prow={},pcol={},target={}").format(act, pid, index, cmd[0],cmd[1], target))
    jump = board.get_enable_jump_d(row, col, cmd[0] - row, cmd[1] - col, check=True)
    if len(jump):
        #print("can jump")
        jumps = list(jump)
        while len(jumps):
            tmpJump = board.get_enable_jump(jumps[0][0], jumps[0][1], True)
            jumps.pop(0)
            if len(tmpJump):
                jumps += tmpJump
                jump += tmpJump
        #print("end search")
        minIdx = 0
        minVal = jump[0][0]
        for i, val in enumerate(jump):
            if val[0] < minVal:
                minIdx = i
                minVal = val[0]
        return pid, jump[minIdx], True

    return pid, cmd, False


def cnvrtMoveCmd2ActCmd(board, id, cmd):
    pid = id
    piece = np.where(board.table==id)
    #print(piece)
    drow = cmd[0] - piece[0][0]
    dcol = cmd[1] - piece[1][0]

    direct = 0
    if drow == 0 and dcol < 0:
        direct = 0
    elif drow < 0 and dcol < 0:
        direct = 1
    elif drow < 0 and dcol == 0:
        direct = 2
    elif drow < 0 and dcol > 0:
        direct = 3
    elif drow == 0 and dcol > 0:
        direct = 4
    else:
        if dcol > 0:
            direct = 5
        elif dcol == 0:
            direct = 6
        else:
            direct = 7

    #print(("direct={}").format(direct))
    act = (pid - 1) * 8 + direct
    return int(act)

class QCNNFunction(chainer.Chain):
    def __init__(self, rows, cols, channels, n_actions, n_hidden_channels=30):
        super(QCNNFunction, self).__init__(
            conv1=L.Convolution2D(channels, n_hidden_channels, ksize=3),
            conv2=L.Convolution2D(n_hidden_channels, n_hidden_channels, ksize=3),
            l1=L.Linear((rows-4) * (cols-4) * n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions)
        )

    def __call__(self, x, test=False):
        s = chainer.Variable(x)
        h = F.leaky_relu(self.conv1(s))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.l1(h))
        h = F.relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(h)


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=81):
        # """
        super(QFunction, self).__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        h = self.l3(h)
        return chainerrl.action_value.DiscreteActionValue(h)

class DoubleDQN():
    def __init__(self, board, isFC=True, gamma=0.8, oneSide=False, opt="MMT", useTrain=True, capacity=10**6, hidden=81):
        self.randAct = RandomActor(board)
        self.board = board
        self.optimizerName = opt
        if opt == "MMT":
            self.optimizer = chainer.optimizers.MomentumSGD()
        elif opt == "Adam":
            self.optimizer = chainer.optimizers.Adam(0.01)
        else:
            self.optimizerName = "SGD"
            self.optimizer = chainer.optimizers.SGD()

        epsilon = 0.1 if useTrain else 0
        self.hidden = hidden
        self.isFC = isFC
        if isFC == True:
            self.q_func = QFunction(board.cols*board.rows, board.cols*8, n_hidden_channels=hidden)
        else:
            self.q_func = QCNNFunction(board.rows, board.cols, 1, board.cols*8)
        self.optimizer.setup(self.q_func)
        self.gamma = gamma

        self.explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1 if useTrain else 0, end_epsilon=epsilon, decay_steps=10 ** 5, random_action_func=self.randAct.random_action_func)

        self.explorer2 = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0, end_epsilon=1.0, decay_steps=10 ** 5, random_action_func=self.randAct.random_action_func)

        self.capacity = capacity
        self.replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=capacity)

        agent_p1 = chainerrl.agents.DoubleDQN(
            self.q_func, self.optimizer, self.replay_buffer, gamma, self.explorer,
            replay_start_size=500, update_interval=1,
            target_update_interval=100)
        agent_p2 = None
        self.oneSide = oneSide
        if oneSide==True:
            agent_p2 = chainerrl.agents.DoubleDQN(
                self.q_func, self.optimizer, self.replay_buffer, gamma, self.explorer2,
                replay_start_size=500, update_interval=1,
                target_update_interval=100)
        else:
            agent_p2 = chainerrl.agents.DoubleDQN(
                self.q_func, self.optimizer, self.replay_buffer, gamma, self.explorer,
                replay_start_size=500, update_interval=1,
                target_update_interval=100)

        self.agents = [agent_p1, agent_p2]

    def act(self):
        x = None
        if self.isFC == False:
            x = self.board.table[np.newaxis, :, :] / self.board.cols
        else:
            x = self.board.table.flatten() / self.board.cols

        action = self.agents[0].act(x)
        target, cmd, jumped = cnvrtActCmd2MoveCmd(self.board, action)
        return target, cmd

    def train(self, n_episodes=10**7, viewStep=100, saveStep=10000):
        print("Training Start")

        xlabel = []
        missPercent = []
        p1WinnerPercent = []
        qValue = []
        move_cnt = 0
        miss = 0
        win = 0
        move_cnt = 0
        p1Winner = 0

        labelName =  self.optimizerName
        if self.isFC == True:
            labelName += "_fc4"
            labelName += "_" + str(self.hidden)
        else:
            labelName += "_cnn4_30"
        labelName += ("_gamma{}").format(self.gamma)
        labelName += ("_capacity{}").format(self.capacity)
        if self.oneSide == True:
            labelName += "_oneSide"

        print(labelName)
        if not os.path.exists(labelName):
            os.mkdir(labelName)

        f = open(labelName + "statistic.csv", 'w')

        for i in range(1, n_episodes):
            # print(("i = {}").format(i))
            self.board.reset()
            reward = 0
            turn = np.random.choice([0, 1])
            last_state = None
            while not self.board.done:
                x = None
                if self.isFC == False:
                    x = self.board.table[np.newaxis, :, :] / self.board.cols
                else:
                    x = self.board.table.flatten() / self.board.cols

                action = self.agents[turn].act_and_train(x, reward)
                target, cmd, jumped = cnvrtActCmd2MoveCmd(self.board, action)
                #
                #board.show()
                self.board.move(target, cmd)
                move_cnt += 1

                #board.show()
                if self.board.done == True:
                    if self.board.winner == 1:
                        reward = 1
                        win += 1
                        if turn == 0:
                            p1Winner += 1
                            # print("win")
                    else:
                        reward = -1
                    if self.board.missed is True:
                        miss += 1
                        # board.show()
                        #print("miss")
                        #test = raw_input(("wait target = {}, cmd = {}").format(target, cmd))
                    if self.isFC == False:
                        x = self.board.table[np.newaxis, :, :] / self.board.cols
                    else:
                        x = self.board.table.flatten() / self.board.cols
                    self.agents[turn].stop_episode_and_train(x, reward, True)

                    if self.agents[(turn + 1) % 2].last_state is not None and self.board.missed is False:
                        if self.isFC == False:
                            x = last_state[np.newaxis, :, :] / self.board.cols
                        else:
                            x = last_state.flatten() / self.board.cols
                        self.agents[(turn + 1) % 2].stop_episode_and_train(x, reward * -1, True)
                else:
                    last_state = self.board.table.copy()

                    jump = []
                    if jumped == True:
                        jump = self.board.get_enable_jump(cmd[0], cmd[1])
                    if len(jump) == 0:
                        self.board.reverse()
                        turn = (turn + 1) % 2
            if i % viewStep == 0:
                msg = "episode:", i, " /move_cnt:", move_cnt, " / rnd:", self.randAct.random_count, " / miss:", miss, " / win:", win, " / p1Winner:", p1Winner, " / statistics:", self.agents[0].get_statistics(), " / epsilon:", self.agents[0].explorer.epsilon
                # print(msg)
                f.write(str(msg))
                f.write("\n")
                f.flush()

                xlabel.append(i)
                missPercent.append(miss * 1.0 / viewStep)
                qValue.append(self.agents[0].get_statistics()[0][1])
                p1WinnerPercent.append(p1Winner * 1.0 / viewStep)

                if len(xlabel) > 100:
                    xlabel.pop(0)
                    missPercent.pop(0)
                    qValue.pop(0)
                    p1WinnerPercent.pop(0)

                plt.clf()
                plt.title(labelName)
                plt.plot(xlabel, missPercent, color="blue")
                plt.plot(xlabel, qValue, color="red")
                plt.plot(xlabel, p1WinnerPercent, color="green")
                plt.pause(0.0001)

                # counter initialization
                miss = 0
                win = 0
                p1Winner = 0
                move_cnt = 0
                self.randAct.random_count = 0
            if i % 10000 == 0:
                fileName = "./" + labelName + "/result_" + str(i)
                self.agents[0].save(fileName)

        print("Training is finished.")
        f.close()

    def load(self, file):
        self.agents[0].load(file)

if __name__ == '__main__':
    board = game.Board()
    board.show()
    dqn = DoubleDQN(board=board,oneSide=False, isFC=True, gamma=0.8, capacity=10**5, opt="MMT", hidden=200)
    dqn.train()
