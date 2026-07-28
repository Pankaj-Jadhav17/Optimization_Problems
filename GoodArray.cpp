#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
class Solution{
    public:
    bool isGood(vector<int>& nums){
        int n = nums.size();
        sort(nums.begin(), nums.end());
        for(int i=0;i<n-1;i++){
            if(nums[i] != i+1){
                return false;
            }
        }
        return nums[n-1] == n-1;
    }
};

int main()
{
   Solution obj;
   int n;
   cout<<"Enter a array size: "<<endl;
   cin >> n;
   //vector<int> nums= {n};
   vector<int> nums(n);

   cout<<"Enter "<< n << " elements"<<endl;
   for(int i=0;i<n;i++){
    cin >> nums[i];
   }

   cout << (obj.isGood(nums) ? "True" : "False") << endl;
//    if (obj.isGood(nums))
//     cout <<"Array is Good"<<endl;
//    else
//     cout <<"Array is not Good"<<endl;
   return 0;
}