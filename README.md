# Inter_IIT_Optimization

### Explaining the Code

So I started with the Heuristics_script code, that is the main code part. 
Then for comparison, I made subsets - Generate_subsets file (9 - 3 of size 30,50,70 each) and ran them using MILP - gurobiMILP file
I compared my heuristics code and the MILP code - both on these subsets and the whole problem.
I saw that the difference in solutions were very different and made some changes iteratively to improve. These were -
1. Initially, priority packages were assigned to the first feasible ULD they encounter, but I changed that so the package will first try to fill the ULDs which already have other priority packages, if that is not feasible, then they will use a new ULD. This resuced the cose 'K' of ULDs with priority packages.
2. The first UID the priority package go in would be the one with the highest weight - this was done again to keep 
3. For economy packages, they were changed s.t. the one with highest delay cost would be filled first instead on just going through in order as given in the file.
4. Compare what is better between taking the penalty vs using a diffferent ULD - the one resulting in minimum added cost will be chosen.
After these changes, I ran both my codes again on actual problem given, and subsets as well, and the results improved by a lot.
The Outputs
1. For heuristics -- Package_Assignment_Output and UID_Utilization
2. For MILP -- MILP_Assignment_Output and MILP_UID_Utilization
