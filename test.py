import bisect
class Solution:
    def medianSlidingWindow(self, nums, k):
        if k == 0: return []
        ans = []
        window = sorted(nums[0:k])
        for i in range(k, len(nums) + 1):
            print(window)
            ans.append((window[k // 2] + window[(k - 1) // 2]) / 2.0)
            if i == len(nums): break
            index = bisect.bisect_left(window, nums[i - k])  # window查找 nums[0] nums[1]等
            print(index)
            window.pop(index)
            bisect.insort_left(window, nums[i])
        return ans
s= Solution()
a= [1,3,-1,-3,5,3,6,7]
ans=s.medianSlidingWindow(a,3)
print(ans)