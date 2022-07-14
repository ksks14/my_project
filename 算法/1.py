# 分割问题
# 1. 若是求和，则一个数字必然对应着它需要的数字，可以对求出每个数字需要的数字。
# 2. 如果条件一已经求出来，对每个数字进行查找就可以了。
#     2.1 缩小查找时间使用字典，利用空间置换时间，字典的查找时间复杂度为o(1)

from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """

        :param nums:
        :param target:
        :return:
        """
        # define a dict
        keys_dict = {}
        for i, num in enumerate(nums):
            key = target - num
            if key in keys_dict:
                return [keys_dict[key], i]
            keys_dict[num] = i
        # no res, return null list
        return []