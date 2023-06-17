# allows using modules defined in any file in project src directory
src_dir = dirname(Base.active_project()) * "/src"
if src_dir ∉ LOAD_PATH
    push!(LOAD_PATH, src_dir)
end

using Initialization: simData, realData, conradData, rosData
using BeliefModels: outputCorMat
using Visualization: visualize
using Exploration: explore

## initialize data for mission
missionData = simData()

## run search alg
@time samples, beliefModel = explore(missionData; visuals=true, sleep_time=.5);

println()
println("Output correlations:")
display(outputCorMat(beliefModel))
