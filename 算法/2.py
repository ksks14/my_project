class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1, l2, ca=0):
        # 1. 循环进行，考虑进位问题
        # 2. 递归进行，考虑进位问题
        val1, val2 = l1.val if l1 else 0, l2.val if l2 else 0
        s = val1 + val2 + ca
        val, ca = s % 10, 1 if s>9 else 0
        next1, next2 = l1.next if l1 else None, l2.next if l2 else None
        if next1 or next2 or ca:
            return ListNode(val, self.addTwoNumbers(next1, next2, ca))
        return ListNode(val)



if __name__ == '__main__':
    a = [1, 2, 3]
    a.insert(0, 1)
    print(3)