import BoardGame
import AIPlayerTest2 as AIPlayer
import HumanPlayer

def test():
    board = BoardGame.Board()
    board.reset()
    board.show()

    human = True

    while True:
        try:
            print(("Select play first:"))
            inCmd = input(("0:Human,1:AI\n"))
            if inCmd==0:
                human = True
            else:
                human = False
            break
        except Exception as e:
            print(e)

    humanPlayer = HumanPlayer.HumanPlayer()
    aiPlayer = AIPlayer.DoubleDQN(board, useTrain=False, hidden=80, isFC=False)
    #aiPlayer = AIPlayer.RandomPlayer(board)

    #loading
    aiPlayer.load("./test")

    while board.done == False:
        board.show()

        target = None
        cmd = None
        if human == True:
            target, cmd = humanPlayer.act(board)
        else:
            target, cmd = aiPlayer.act()

        print(target)
        print(cmd)
        while True:
            pos = board.get_target(target)
            board.move(target, cmd)
            board.show()
            if board.winner!=None:
                if board.missed:
                    print("missed!")
                    human = not(human)
                print(("---Winner = {}---").format("You!" if human else "AI"))
                break

            if human == False:
                board.reverse()
                break
            if abs(pos[0]-cmd[0])>1 or abs(pos[1]-cmd[1])>1:
                jump = board.get_enable_jump(cmd[0],cmd[1])
                if len(jump)!=0:
                    print("You can move once more.")
                    cmd = humanPlayer.cmd(jump)
                    if cmd!=None:
                        continue
            board.reverse()
            break
        human = not human

if __name__ == '__main__':
    test()
