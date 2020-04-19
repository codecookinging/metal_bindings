import bisect
class Solution:
    def sumCal(self , num , times ):
        def get_sum(num,times):
            total = 0
            for i in range(times):
                total+=num*pow(10,i)
            return total
        # write code here
        dp = [[0]*(times+1) for i in range(num+1)]

        for k in range(num+1):
            dp[k][1] = k
        print(dp)
        for i in range(1,num+1):
            for j in range(1,times+1):
                temp = get_sum(i,j)
                dp[i][j] = dp[i][j-1]+temp
        return dp[-1][-1]
s= Solution()
re = s.sumCal(3,2)
print(re)