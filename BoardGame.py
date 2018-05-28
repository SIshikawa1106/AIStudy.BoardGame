import numpy as np
from collections import namedtuple
import random

Piece = namedtuple('Piece', ('PieceID', 'cpos', 'npos'))


class Board():
    def __init__(self, rows=8, cols=5):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self):
        self.table = np.array([0]*self.rows*self.cols, dtype=np.float32).reshape(self.rows, self.cols)
        self.tmpTalbe = self.table.copy()
        self.winner = None
        self.missed = False
        self.done = False
        
        rowStr= ""
        hr = ""
        for i in range(self.cols):
            if i!=0:
                rowStr = rowStr + "|"
            else:
                hr = "\n"
                
            rowStr = rowStr + " {} "
            hr = hr + "------"
            self.table[0][i] = -1 * (self.cols-i)
            self.table[self.rows-1][i] = i+1
            
        hr = hr + "\n"
        self.strBoard = ""

        for i in range(self.rows):
            self.strBoard = self.strBoard + rowStr
            if i!=self.rows-1:
                self.strBoard = self.strBoard + hr
        

    def reverse(self):
        self.table = np.flipud(self.table)
        self.table = np.fliplr(self.table)
        self.table = self.table*-1
        self.tmpTalbe.fill(0)
        #print(self.table)
        
        
    def check_winner(self):

        num1 = np.where(self.table[0]>0)[0]
        num2 = np.where(self.table[1]>0)[0]
        
        if len(num1)>0 or len(num2)>0:
            self.winner = 1
            self.done = True
            

    def check_enable_range(self, row, col):
        
        if row<0 or col<0 or col>=self.cols or row>=self.rows:
            return False
        return True

    def get_enable_step(self, row, col):
        enable = []
        
        for y in range(-1, 2):
            for x in range(-1, 2):
                prow = y+row
                pcol = x+col
                if self.check_enable_range(prow, pcol)==False:
                    continue
                if prow==0 and pcol==0:
                    continue

                if self.table[prow][pcol]==0:
                    enable.append([prow,pcol])

        #print(("step enable = {}").format(enable))
        return enable

    def get_enable_jump_d(self, row, col, drow, dcol, check=False):
        if check:
            self.tmpTalbe[row][col] = 2
        enable = []
        dist = 1
        found = False
        while found == False:
            #print(dist)
            prow = (drow*dist) + row
            pcol = (dcol*dist) + col

            #print(("row={},col={},rows={},cols={}").format(prow,pcol,self.rows,self.cols))
            if self.check_enable_range(prow, pcol)==False:
                break

            if self.table[prow][pcol]!=0:
                for dist2 in range(dist+1, dist*2+1):
                    tprow = (drow*dist2) + row
                    tpcol = (dcol*dist2) + col
                    if self.check_enable_range(tprow, tpcol)==False:
                        return enable
                    if self.table[tprow][tpcol]!=0:
                        return enable
                    if self.tmpTalbe[tprow][tpcol]!=0:
                    	return enable
                else:
                    enable.append([tprow,tpcol])
                    if check:
                        self.tmpTalbe[tprow][tpcol] = 2
                    return enable
            dist += 1
        return enable

    def get_enable_jump(self, row, col, check=False):
        enable = []

        for y in range(-1, 2):
            for x in range(-1, 2):

                if y==0 and x==0:
                    continue

                tmp = self.get_enable_jump_d(row, col, y, x, check)
                if len(tmp)!=0:
                    enable = enable + tmp
                    
        #print(("row={}, col={}, move point={}").format(row, col, enable))
        return enable

    def get_target_all(self):
        #print(self.table)
        candidate = np.where(self.table>0)

        target = []
        if len(candidate)==0:
            print("get enable target has error.")
            return target

        #print(("candidate={}").format(len(candidate[0])))
        
        ids = []
        
        for i in range(len(candidate[0])):
            target.append([candidate[0][i], candidate[1][i]])
            ids.append(self.table[candidate[0][i]][candidate[1][i]])

        return ids, target

    def get_target(self, index):
        #print(("get_target input index = {}").format(index))
        #print(self.table)
        tmp = np.where(self.table==index)
        
        return [tmp[0][0],tmp[1][0]]
    
    def get_enable_pieces(self):
        #print(self.table)

        ids, targets = self.get_target_all()
        output = []
        
        for index in range(len(targets)):
            tmpCmd1 = self.get_enable_step(targets[index][0], targets[index][1])
            tmpCmd2 = self.get_enable_jump(targets[index][0], targets[index][1])
            cmd = []
            
            if len(tmpCmd1)!=0:
                cmd = tmpCmd1
            if len(tmpCmd2)!=0:
                cmd = cmd + tmpCmd2

            if len(cmd)!=0:
                output.append(Piece(PieceID=ids[index],cpos=[targets[index][0], targets[index][1]], npos=cmd))

        return output    

    def move(self, id, command):
        #print("move")
        target = self.get_target(id)
        
        if self.check_enable_range(command[0], command[1])==False or self.table[command[0]][command[1]]!=0:
            self.winner = -1
            self.missed = True
            self.done = True
            return False
            
        self.table[target[0]][target[1]] = 0
        self.tmpTalbe[target[0]][target[1]] = 1
        self.table[command[0]][command[1]] = id

        self.check_winner()

        return True

    def show(self):

        temptable = []
        for row in self.table:
            for col in row:
                if col<0:
                    temptable.append(" x ")
                elif col>0:
                    temptable.append(("({})").format(int(col)))
                else:
                    temptable.append("   ")

        print("\n================BOARD================")
        print((self.strBoard).format(*temptable))
        
            
        
        
        
