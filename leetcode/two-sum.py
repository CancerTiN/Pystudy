# class List(list):
#     pass

# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         end = len(nums)
#         for i, a in enumerate(nums):
#             for j in range(i + 1, end):
#                 b = nums[j]
#                 if a + b == target:
#                     return [i, j]

# class Solution:
#     def twoSum(self, nums, target):
#         hashmap = {}
#         for index, num in enumerate(nums):
#             another_num = target - num
#             if another_num in hashmap:
#                 return [hashmap[another_num], index]
#             hashmap[num] = index
#         return None

class Solution:
    def twoSum(self, nums, target: int):
        num_to_index = dict()
        for index, num in enumerate(nums):
            another_num = target - num
            another_index = num_to_index.get(another_num)
            if another_index is not None:
                return [another_index, index]
            num_to_index[num] = index


if __name__ == '__main__':
    s = Solution()
    r = s.twoSum([2, 7, 11, 15], 9)
    print(r)
