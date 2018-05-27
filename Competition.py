import BoardGame
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
            inCmd = input(("0:Human,1:AI\n"))
            if inCmd==0:
                human = True
            else:
                human = False
            break
        except Exception as e:
            print(e)

    humanPlayer = HumanPlayer.HumanPlayer()
    aiPlayer = AIPlayer.DoubleDQN(board, useTrain=False)

    #loading
    aiPlayer.load("./MMT_fc4_gamma0.8/result_100000")

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
            board.move(target, cmd)
            board.show()
            if board.winner!=None:
                print(("---Winner = {}---").format("You!" if human else "AI"))
                break

            if human == False:
                board.reverse()
                break
            jump = board.get_enable_jump(cmd[0],cmd[1])
            if len(jump)==0:
                board.reverse()
                break
            else:
                print("You can move once more.")
                cmd = humanPlayer.cmd(jump)
        human = not human

if __name__ == '__main__':
    test()
