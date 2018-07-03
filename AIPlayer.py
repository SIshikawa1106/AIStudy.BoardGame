import BoardGame as game
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from Memory import Memory

pieceTrain = True
moveTrain = True

class RandomActor:
    def __init__(self, board):
        self.board = board
        self.random_count = 0
        self.pieceId = -1
        self.random_count = 0

    def random_act_piece_func(self):
        self.random_count += 1
        #ids, targets = self.board.get_target_all()

        pieces = self.board.get_enable_pieces()

        sum = 0
        for p in pieces:
            val = p.cpos[0] + 0.1
            if p[0]<=1:
                val = 0.1
            sum += val


        rand = random.random() * sum

        sum = 0
        for i in range(len(pieces)):
            val = pieces[i].cpos[0] + 0.1
            if pieces[i].cpos[0]<=1:
                val = 0.1
            sum += val
            if sum>= rand:
                self.pieceId = int(pieces[i].PieceID)
                return int(self.pieceId) - 1
        self.pieceId = int(pieces[-1].PieceID)
        return self.pieceId - 1

    def set_piece(self, pieceId):
        if pieceId<=0 or pieceId>self.board.cols:
            raise ValueError
        self.pieceId = pieceId

    def random_move(self):
        pos = self.board.get_target(self.pieceId)

        next_pos = None
        if self.board.isMoved()==False:
            next_pos = self.board.get_enable_step(pos[0],pos[1])

        next_pos2 = self.board.get_enable_jump(pos[0],pos[1])

        if len(next_pos2)!=0:
            if next_pos==None:
                next_pos = next_pos2
            else:
                next_pos = next_pos + next_pos2
        if self.board.isMoved():
            #print(next_pos)
            #print(pos)
            if next_pos==None:
                next_pos = [pos]
            else:
                next_pos = next_pos + [pos]

        sum = 0
        for p in next_pos:
            val = self.board.rows-p[0] + 1
            sum += val*val
            #print(("p={},w={}").format(p, val))

        rand = random.random() * sum
        #print(("rand={},sum={}").format(rand, sum))

        sum = 0
        next_cmd = None
        for i in range(len(next_pos)):
            val = self.board.rows - next_pos[i][0] + 1
            sum += val*val
            #print(("p={},sum={}").format(next_pos[i], sum))
            if sum>= rand:
                next_cmd = next_pos[i]
                break
        if next_cmd == None:
            next_cmd = next_pos[-1]

        #print(("random cmd = {}").format(next_cmd))
        return cnvtMoveCmd2ActCmd(self.board, self.pieceId, next_cmd)

def cnvrtActCmd2MoveCmd(board, id, cmdId):

    target = board.get_target(id)
    index = cmdId % 8
    jump = cmdId // 8

    row = target[0]
    col = target[1]
    #print(("target = {}").format(target))
    if jump == 2:
        cmd = [0,0]
    elif index == 0:
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

    #print(("pos=({},{}),cmd=({},{})").format(row,col, cmd[0],cmd[1]))
    if jump == 1:
        tmp = board.get_enable_jump_d(row, col, cmd[0]-row, cmd[1]-col)
        if len(tmp)==0:
            cmd = None
        else:
            cmd = tmp[0]
        #print(("jump={}").format(cmd))

    #print(("cnvrtActCmd2MoveCmd:{},{},{}").format(id, cmdId, cmd))
    return jump==1, int(id), cmd


def cnvtDirect2Index(drow, dcol):
    direct = 0
    if drow==0 and dcol==0:
        direct = 16
    elif drow == 0 and dcol < 0:
        direct = 0 if dcol==-1 else 8
    elif drow < 0 and dcol < 0:
        direct = 1 if drow==-1 else 9
    elif drow < 0 and dcol == 0:
        direct = 2 if drow==-1 else 10
    elif drow < 0 and dcol > 0:
        direct = 3 if drow==-1 else 11
    elif drow == 0 and dcol > 0:
        direct = 4 if dcol==1 else 12
    else:
        if dcol > 0:
            direct = 5 if dcol==1 else 13
        elif dcol == 0:
            direct = 6 if drow==1 else 14
        else:
            direct = 7 if drow==1 else 15

    return int(direct)

def cnvtMoveCmd2ActCmd(board, id, cmd):
    pid = id
    piece = np.where(board.table==id)
    #print(piece)
    drow = cmd[0] - piece[0][0]
    dcol = cmd[1] - piece[1][0]

    #print(("pos={},cmd={},dpos={}").format([piece[0][0],piece[1][0]],cmd, [drow,dcol]))
    direct = cnvtDirect2Index(drow, dcol)

    #print(("direct={}").format(direct))

    return int(direct)

class PieceModel(chainer.Chain):
    def __init__(self, rows, cols, channels, n_actions, n_hidden_channels=30):
        super(PieceModel, self).__init__(
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
        #h = F.sigmoid(self.l2(h))
        h = self.l2(h)
        return chainerrl.action_value.DiscreteActionValue(h)

class Model(chainer.Chain):
    def __init__(self, rows, cols, channels, n_actions, n_hidden_channels=30):
        super(Model, self).__init__(
            conv1=L.Convolution2D(channels, n_hidden_channels, ksize=5),
            conv2=L.Convolution2D(n_hidden_channels, n_hidden_channels, ksize=3),
            l1=L.Linear((rows-6) * (cols-6) * n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions)
        )

    def __call__(self, x, test=False):
        s = chainer.Variable(x)
        h = F.leaky_relu(self.conv1(s))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.l1(h))
        #h = F.sigmoid(self.l2(h))
        h = self.l2(h)
        return chainerrl.action_value.DiscreteActionValue(h)



class Agent():
    def __init__(self, board, gamma=0.9, capacity=10**6, pmodel=None, amodel=None, memory=None, view=False, p_epsilon=(1.0,1.0), m_epsilon=(0.1,0.1), gpu=0):
        self.randomAct = RandomActor(board)
        self.board = board
        self.lastAct = None
        self.memory = memory
        self.decay_steps = 10 ** 7
        self.gpu = gpu
        self.view = view

        self.explorer_p = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=p_epsilon[0], end_epsilon=p_epsilon[1], decay_steps=self.decay_steps,
            random_action_func=self.randomAct.random_act_piece_func)


        self.explorer_m = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=m_epsilon[0], end_epsilon=m_epsilon[1], decay_steps=self.decay_steps,
            random_action_func=self.randomAct.random_move)

        hidden = 64
        batch_size = 128

        if pmodel==None:
            self.pieceQFunc = PieceModel(board.rows, board.cols, 1, board.cols, hidden)
        else:
            self.pieceQFunc = pmodel
        if amodel==None:
            self.moveQFunc = Model(board.rows*2-1, board.cols*2 - 1, 1+9+8, 9+8, hidden)
        else:
            self.moveQFunc = amodel

        self.optimizerPiece = chainer.optimizers.SGD()
        #self.optimizerPiece = chainer.optimizers.MomentumSGD()
        #self.optimizerPiece = chainer.optimizers.Adam()
        self.optimizerPiece.setup(self.pieceQFunc)
        self.optimizerMove = chainer.optimizers.SGD()
        #self.optimizerMove = chainer.optimizers.MomentumSGD()
        #self.optimizerMove = chainer.optimizers.Adam()
        #self.optimizerMove = chainer.optimizers.RMSprop()
        self.optimizerMove.setup(self.moveQFunc)

        self.replay_buffer_p = chainerrl.replay_buffer.ReplayBuffer(capacity=capacity)
        self.replay_buffer_m = chainerrl.replay_buffer.ReplayBuffer(capacity=capacity)

        self.pieceAgent = chainerrl.agents.DoubleDQN(
            self.pieceQFunc, self.optimizerPiece, self.replay_buffer_p,
            gamma, self.explorer_p, gpu=self.gpu,
            replay_start_size=1000, update_interval=1,
            target_update_interval=1000, minibatch_size=batch_size)

        self.moveAgent = chainerrl.agents.DoubleDQN(
            self.moveQFunc, self.optimizerMove, self.replay_buffer_m,
            gamma, self.explorer_m, gpu=self.gpu,
            replay_start_size=1000, update_interval=1,
            target_update_interval=1000, minibatch_size=batch_size)

    def act(self, iact=None):
        act = 0
        if iact==None:
            x = (self.board.table/self.board.cols)[np.newaxis,]
            act = self.pieceAgent.act(x) + 1
            #act = self.randomAct.random_act_piece_func()
        else:
            act =iact

        self.lastAct = act
        map = self._make_input_data()

        move = self.moveAgent.act(map)
        if self.board.isMoved()==True and move<8:
            move = 16
        #print("move={}".format(move))
        return cnvrtActCmd2MoveCmd(self.board, act, move)

    def _make_input_data(self):

        pos = self.board.get_target(self.lastAct)
        map = np.zeros((self.board.rows*2-1, self.board.cols*2-1), dtype=np.float32)
        map.fill(-1)
        cy = map.shape[0]//2
        cx = map.shape[1]//2
        tmp = np.where(self.board.table>0, 1, self.board.table)
        tmp = np.where(self.board.table<0, -1, tmp)
        map[cy-pos[0]:cy-pos[0]+self.board.rows, cx-pos[1]:cx-pos[1]+self.board.cols] = tmp

        actMap = np.zeros((8+8+1, *map.shape), dtype=np.float32)
        enableCmds= self.board.get_enable_cmd(pos[0], pos[1])

        for cmd in enableCmds:
            actId = cnvtMoveCmd2ActCmd(self.board, self.lastAct, cmd)
            actMap[actId,:,:] = 1

        if self.board.isMoved()==True:
            actMap[-1,:,:] = 1

        map = map[np.newaxis,]
        map = np.concatenate((map, actMap), axis=0)
        
        return map

    def act_and_train(self, reward, iact=None):
        act = 0
        if iact==None:
            x = (self.board.table/self.board.cols)[np.newaxis,]
            if pieceTrain:
                act = self.pieceAgent.act_and_train(x, reward) + 1
            else:
                act = self.pieceAgent.act(x) + 1
            #act = self.randomAct.random_act_piece_func() + 1
            if self.memory is not None:
                self.memory.setState(state=self.board.table, action=act, rwd=reward)
        else:
            act =int(iact)
        #print(("act={}").format(act))

        self.lastAct = act
        map = self._make_input_data()
        move = self.moveAgent.act_and_train(map, reward)
        #print(move)
        if self.board.isMoved()==True and move<8:
            return True, act, None

        #print(("act={}, move={}").format(act, move))
        return cnvrtActCmd2MoveCmd(self.board, act, move)


    def stop_episode_and_train(self, reward):
        if self.lastAct is None:
            return
        if self.memory is not None:
            self.memory.endState(self.board.table, reward)

        if pieceTrain:
            x = self.board.table / self.board.cols
            self.pieceAgent.stop_episode_and_train(x[np.newaxis,], reward, True)

        if self.lastAct is not None:
            map = self._make_input_data()
            move = self.moveAgent.stop_episode_and_train(map, reward, True)

        self.lastAct = None

    def save(self, file):
        if self.gpu!=None:
            self.pieceAgent.model.to_cpu()
            self.moveAgent.model.to_cpu()
        self.pieceAgent.save(file+"/piece")
        self.moveAgent.save(file + "/move")
        if self.gpu!=None:
            self.pieceAgent.model.to_gpu(self.gpu)
            self.moveAgent.model.to_gpu(self.gpu)

    def load(self, file):
        if os.path.isdir(file + "/piece"):
            self.pieceAgent.load(file+"/piece")
        if os.path.isdir(file + "/move"):
            self.moveAgent.load(file + "/move")

        if self.gpu!=None:
            self.pieceAgent.model.to_gpu(self.gpu)
            self.moveAgent.model.to_gpu(self.gpu)

if __name__ == '__main__':
    print("Train")
    board = game.Board()
    board.show()
    memory = None
    agent1 = Agent(board, gamma=0.8, memory=memory, m_epsilon=(0.1,0.1), p_epsilon=(1.0, 0.1))
    agent2 = Agent(board, gamma=0.8, amodel=agent1.moveQFunc, pmodel=agent1.pieceQFunc, m_epsilon=(1.0,0.1), p_epsilon=(1.0, 0.1))

    #"""
    agent1.load("./test/")
    agent2.load("./test/")
    #"""
    agents = [agent1,agent2]
    print("Training Start")

    xlabel = []
    missPercent = []
    p1WinnerPercent = []
    qValue = []
    qValue2 = []
    move_cnt = 0
    miss = 0
    win = 0
    move_cnt = 0
    p1Winner = 0
    viewStep = 50
    view = False

    labelName = "Test_SGD_64"
    print(labelName)
    if not os.path.exists(labelName):
        os.mkdir(labelName)

    f = open("./" + labelName + "/statistic.csv", 'w')

    fig = plt.figure()
    ax1 = plt.subplot()
    ax2 = ax1.twinx()  # 2つのプロットを関連付ける

    #for i in range(1, 10*10):
    i = 0
    while(True):
        i+=1
        # print(("i = {}").format(i))
        board.reset()
        reward = 0
        turn = np.random.choice([0, 1])
        last_state = None
        preAct = None
        while not board.done:
            jumped, target, cmd = agents[turn].act_and_train(reward, preAct)
            preAct = target
            if view:
                print(jumped)
                print(target)
                print(cmd)
                board.show()

            board.move(target, cmd)
            if view:
                board.show()
            move_cnt += 1

            if board.done == True:
                if board.winner == 1:
                    reward = 1
                    win += 1
                    if turn == 0:
                        p1Winner += 1
                        print("win")
                else:
                    reward = -1
                if board.missed is True:
                    miss += 1
                    # board.show()
                    # if turn != 0:
                    print("miss")
                    # test = raw_input(("wait target = {}, cmd = {}").format(target, cmd))
                agents[turn].stop_episode_and_train(reward)

                board.reverse()
                if last_state is not None and board.missed is False:
                    agents[(turn + 1) % 2].stop_episode_and_train(reward * -1)
                else:
                    agents[(turn + 1) % 2].stop_episode_and_train(0)
            else:
                last_state = board.table.copy()

                jump = []
                if jumped == True:
                    jump = board.get_enable_jump(cmd[0], cmd[1])
                if len(jump) == 0:
                    board.reverse()
                    preAct = None
                    turn = (turn + 1) % 2
                    if view:
                        print("*********turn change*********")
                elif view:
                    print("*********once more*********")
            #test = input()

        if i % viewStep == 0:
            msg = "episode:", i, " /move_cnt:", move_cnt, " / miss:", miss, " / win:", win, " / p1Winner:", p1Winner,\
                  " / statistics(piece):", agents[0].pieceAgent.get_statistics(), \
                  " / statistics(move):", agents[0].moveAgent.get_statistics(), \
                  " / epsilon(piece):", agents[0].explorer_p.epsilon,\
                  " / epsilon(move):", agents[0].explorer_m.epsilon
            # print(msg)
            f.write(str(msg))
            f.write("\n")
            f.flush()

            xlabel.append(i)
            missPercent.append(miss * 1.0 / viewStep)
            qValue.append(agents[0].moveAgent.get_statistics()[0][1])
            qValue2.append(agents[0].pieceAgent.get_statistics()[0][1])
            p1WinnerPercent.append(p1Winner * 1.0 / viewStep)

            if len(xlabel) > 100:
                xlabel.pop(0)
                missPercent.pop(0)
                qValue.pop(0)
                qValue2.pop(0)
                p1WinnerPercent.pop(0)

            #plt.clf()
            plt.title(labelName)
            ax1.plot(xlabel, missPercent, color="blue")
            ax1.plot(xlabel, p1WinnerPercent, color="green")
            ax2.plot(xlabel, qValue, color="red")
            ax2.plot(xlabel, qValue2, color='c')
            plt.pause(0.0001)

            # counter initialization
            miss = 0
            win = 0
            p1Winner = 0
            move_cnt = 0
        if i % 1000 == 0:
            fileName = "./" + labelName + "/result_" + str(i)
            agents[0].save(fileName)

    print("Training is finished.")
    f.close()



