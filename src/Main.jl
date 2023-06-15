# allows using modules defined in any file in project src directory
src_dir = dirname(Base.active_project()) * "/src"
if src_dir ∉ LOAD_PATH
    push!(LOAD_PATH, src_dir)
end

using Initialization: simData, realData, conradData, rosData
using BeliefModels: outputCorMat
using Visualization: visualize
using Exploration: explore

## initialize region

# initialize data use simData or realData for this
region, start_loc, weights, num_samples, prior_samples = simData()

## run search alg
@time samples, beliefModel = explore(region, start_loc, weights;
                                     num_samples,
                                     prior_samples,
                                     visuals=true,
                                     sleep_time=0.0);

println()
println("Output correlations:")
display(outputCorMat(beliefModel))
