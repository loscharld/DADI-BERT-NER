#!/usr/bin/env python 
# -*- coding:utf-8 -*-

class node(object):

    def __init__(self):
        self.next = {}       #相当于指针，指向树节点的下一层节点
        self.fail = None     #失配指针，这个是AC自动机的关键
        self.isWord = False  #标记，用来判断是否是一个标签的结尾
        self.word = ""       #用来储存标签


class ac_automation(object):

    def __init__(self):
        self.root = node()

    def add(self, word):
        temp_root = self.root
        for char in word:
            if char not in temp_root.next:
                temp_root.next[char] = node()
            temp_root = temp_root.next[char]
        temp_root.isWord = True
        temp_root.word = word

    def make_fail(self):
        temp_que = []
        temp_que.append(self.root)
        while len(temp_que) != 0:
            temp = temp_que.pop(0)
            p = None
            for key,value in temp.next.item():
                if temp == self.root:
                    temp.next[key].fail = self.root
                else:
                    p = temp.fail
                    while p is not None:
                        if key in p.next:
                            temp.next[key].fail = p.fail
                            break
                        p = p.fail
                    if p is None:
                        temp.next[key].fail = self.root
                temp_que.append(temp.next[key])

    def search(self, content):
        p = self.root
        result = []
        currentposition = 0

        while currentposition < len(content):
            word = content[currentposition]
            while word in p.next == False and p != self.root:
                p = p.fail

            if word in p.next:
                p = p.next[word]
            else:
                p = self.root

            if p.isWord:
                result.append(p.word)

            currentposition += 1
        return result

if __name__=='__main__':
    ac = ac_automation()
    ac.add('奥迪A6')
    ac.add('奥迪A6L')
    ac.add('奥迪')
    # ac.add('生存游戏')
    text='奥迪A6L 基本型 2000.0 ANQ 1.8L 自然吸气 汽油 手动变速器(MT) 5'
    print(ac.search(text))