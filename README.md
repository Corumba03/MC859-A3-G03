# MC859-A3-G03
Repositório da A3 do grupo 03

How to run the solver:

1) Set the following parameters by keyword on main.py

obj_function: the function to be evaluated
tenure: size of the Tabu list, leave as 0 for a dynamic size
no_improv_iter: number of iterations without improvement before stopping
max_iter: number of iterations before stopping 
maximize (optional): False if you want to minimize the function, True otherwise (it is True by default)
constructive_type: one of ['std']
search_type: Choose between 'first' (for first improving) or 'best' (for best improving)
tabu_check: Choose between 'strict' (to tabu any moves using the element tabu) or 'relaxed' (to tabu only the specific move used)


2) Set the 'minutes' variable in runner.py to specify the maximum amount of minutes before a timeout

3) Put the all the instances on the 'instances' folder, run runner.py then check the results on the folder 'logs' that will be generated