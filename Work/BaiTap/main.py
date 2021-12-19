# Khai báo thư viện cần dùng: cách cài pip install cvxopt hoặc conda install cvxopt
from cvxopt import matrix, solvers

'''     Bài 1 Các điều kiện dàng buộc có thể viết thành
        G = [ -1, 1 1, 1 0, 1 1, -2 -1, 0 0, -1]
        h = [ 1, 2, 0, 4, 0, 0 ]                            '''
def Bai1():
    c = matrix([2., 1.])
    G = matrix([[-1., 1., 1., 1., 0., 1.], [1., -2., -1., 0., 0., -1.]])
    h = matrix([1., 2., 0., 4., 0., 0.])
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h)
    print("Solution",sol['x'])

'''
    Bài 2 Các điều kiện dàng buộc có thể viết thành c = [-5, -10, -15, -4] 
    G = [1, 0, 1, 0 0, 1, 0, 1 1, 1, 0, 0 0, 0, 1, 1 -1, 0, 0, 0 0, -1, 0, 0 0, 0, -1, 0 0, 0, 0, -1] 
    h = [600, 400, 800, 700, 0, 0, 0, 0]
'''
def Bai2():
    c = matrix([-5., -10., -15., -4.])
    G = matrix([[1., 0., 1., 0., -1., 0., 0., 0.],
                [0., 1., 1., 0., 0., -1., 0., 0.],
                [1., 0., 0., 1., 0., 0., -1., 0.],
                [0., 1., 0., 1., 0., 0., 0., -1.]])
    h = matrix([600., 400., 800., 700., 0., 0., 0., 0.])
    solvers.options['show_progress'] = True
    sol = solvers.lp(c, G, h)
    print("Solution", sol['x'])

'''
    Bài 3 : Các điều kiện dàng buộc có thể viết thành 
                P = [1 0 0 1] 
                q = [1 1] 
                G = [0 1 1 0 1 1 -1 0 0 -1] 
                H = [0 0 1 0 0]         
'''
def Bai3():
    P = matrix([[1., 0.],
                [0., 1.]])
    q = matrix([1., 1.])
    G = matrix([[0., 1., 1., -1., 0.], [1., 0., 1., 0., -1.]])
    h = matrix([0., 0., 1., 0., 0])
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    print("Solution:\n", sol['x'])

if __name__ == '__main__':
    print("Kết quả của bài 1:");
    Bai1()
    print("\nKết quả của bài 2:");
    Bai2()
    print("\nKết quả của bài 3:");
    Bai3()