import BoardGame
#import AIPlayerTest2 as AIPlayer
import AIPlayer
import HumanPlayer

def test():
    board = BoardGame.Board()
    board.reset()
    board.show()

    human = True

    while True:
        try:
            print(("Select play first:"))
            inCmd = int(input(("0:Human,1:AI\n")))
            if inCmd==0:
                human = True
            else:
                human = False
            break
        except Exception as e:
            print(e)

    humanPlayer = HumanPlayer.HumanPlayer()
    #aiPlayer = AIPlayer.DoubleDQN(board, useTrain=False, hidden=80, isFC=False)
    aiPlayer = AIPlayer.Agent(board, gpu=None, p_epsilon=(1.0,1.0), m_epsilon=(0,0))
    #aiPlayer = AIPlayer.RandomPlayer(board)

    #loading
    aiPlayer.load("./test/")
    preAct = None
    while board.done == False:
        if human == True:
            print("Human Turn")
        else:
            print("AI Turn")
        board.show()

        target = None
        cmd = None
        if human == True:
            target, cmd = humanPlayer.act(board)
        else:
            jumped, target, cmd = aiPlayer.act()

        assert target > 0
        print("target={}".format(target))
        print("cmd={}".format(cmd))
        while True:
            print("move")
            pos = board.get_target(target)
            print(board.move(target, cmd))
            board.show()
            if board.winner!=None:
                if board.missed:
                    print("missed!")
                    human = not(human)
                print(("---Winner = {}---").format("You!" if human else "AI"))
                break
            tmp = input()
            if abs(pos[0]-cmd[0])>1 or abs(pos[1]-cmd[1])>1:
                jump = board.get_enable_jump(cmd[0],cmd[1])
                if len(jump)!=0 and human==True:
                    print("You can move once more.")
                    cmd = humanPlayer.cmd(jump)
                    if cmd!=None:
                        continue
                elif len(jump)!=0:
                    preAct = target
                    continue
            else:
                print("change")
            preAct = None
            board.reverse()
            break
        human = not human

if __name__ == '__main__':
    test()
