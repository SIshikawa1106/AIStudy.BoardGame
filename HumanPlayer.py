import BoardGame
import numpy as np

class HumanPlayer:
    def act(self, board):
        valid = False
        while not valid:
            try:
                pieces = board.get_enable_pieces()

                act = input(("Select move target id.({}-{}): ").format(1, board.cols))
                act = int(act)
                if act<1 or act>board.cols:
                    print("Invalid target")
                    continue
                print("Select move point")
                for piece in pieces:
                    if piece.PieceID == act:
                        for i, cmd in enumerate(piece.npos):
                            print(("i={}:{}").format(i, cmd))
                        cmd = input("Input(i=):")
                        cmd = int(cmd)
                        if cmd<0 or cmd >= len(piece.npos):
                            print("Invalid command")
                            break
                        else:
                            valid = True
                            return act, piece.npos[cmd]
                            
            except Exception as e:
                print(e)
                
    def cmd(self, jump):
        valid = False

        while not valid:
            try:
                print("Select move point")
                for i, cmd in enumerate(jump):
                	print(("i={}:{}").format(i, cmd))
                	
                cmd = input("Input(i=, i==-1 means skip):")
                cmd = int(cmd)
                if cmd<-1 or cmd >= len(jump):
                    print("Invalid command")
                    continue
                if cmd==-1:
                    print("Skipe")
                    return None
                valid = True
                return jump[cmd]
            except Exception as e:
                print(e)
                

def test():
    board = BoardGame.Board()
    board.reset()
    board.show()

    pieces = board.get_enable_pieces()
    for piece in pieces:
        print(piece)

    human = HumanPlayer()
    while board.done == False:
        board.show()
        target, cmd = human.act(board)
        pos = board.get_target(target)
        print(target)
        print(cmd)
        while True:
            print(board.isMoved())
            board.move(target, cmd)
            print(board.isMoved())
            board.show()
            if board.winner!=None:
                print(("---Winner = {}---").format(board.winner))
                break
            if abs(pos[0]-cmd[0])>1 or abs(pos[1]-cmd[1])>1:
                jump = board.get_enable_jump(cmd[0],cmd[1])
                if len(jump)!=0:
                    print("You can move once more.")
                    cmd = human.cmd(jump)
                    if cmd!=None:
                    	continue
            board.reverse()
            break

if __name__ == '__main__':
    test()
