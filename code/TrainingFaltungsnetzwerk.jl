# Festlegen der Trainingsmethode und -parameter
method = SGD()
train_params = make_solver_parameters(method, max_iter=10000),
regu_coef=0.0, mom_policy=MomPolicy.Fixed(0.0),
lr_policy=LRPolicy.Fixed(0.001), load_from="VGGsnapshots")
solver = Solver(method, train_params)

# Hinzufügen von Coffebreaks für Kontrolle des Trainings / Speichern der Trainingsstände
add_coffee_break(solver, TrainingSummary(), every_n_iter=1000)
add_coffee_break(solver, Snapshot(VGGsnapshots), every_n_iter=2500)

# Trainieren des Netzes
solve(solver, netDef)
