#include<bits/stdc++.h>
using namespace std;
int solve(int day, int activity,vector<vector<int>> &points, int n){
    
    if(day == 0){
        return max(max(points[day][0], points[day][1]), points[day][2]);
    }
    if(day == n - 1){
        int choice1 = points[day][0] + solve(day-1,0,points,n);
        int choice2 = points[day][1] + solve(day-1,1,points,n);
        int choice3 = points[day][2] + solve(day-1,2,points,n);
        return max(max(choice1,choice2), choice3);
    }
    int activitiesPossible[2];
    if (activity == 0) {
        activitiesPossible[0] = 1;
        activitiesPossible[1] = 2;
    }
    else if (activity == 1) {
        activitiesPossible[0] = 0;
        activitiesPossible[1] = 2;
    }
    else if (activity == 2) {
        activitiesPossible[0] = 0;
        activitiesPossible[1] = 1;
    }
    //only two choices if this is not the last day
    int choice1 = points[day][activitiesPossible[0]] + solve(day-1,activitiesPossible[0],points,n);
    int choice2 = points[day][activitiesPossible[1]] + solve(day-1,activitiesPossible[1],points,n);
    return max(choice1,choice2);
}


int ninjaTraining(int n, vector<vector<int>> &points)
{
    return solve(n-1, 0, points, n);
    // Write your code here.
    //points[i][j] indicates ith day, jth activity 
    //1 < i < n, 1 < j < 3

}
int main() {
    int t;
    cin >> t;
    while(t--){
      int n;
      cin >> n;
      vector<vector<int>> points(n, vector<int>(3));
      for (int i = 0; i < n; i++) {
          for (int j = 0; j < 3; j++) {
              cin >> points[i][j];
          }
      }
      cout << ninjaTraining(n, points) << endl;
    }
    
      return 0;
}
