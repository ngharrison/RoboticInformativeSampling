# allows using modules defined in any file in project src directory
src_dir = dirname(Base.active_project()) * "/src"
if src_dir ∉ LOAD_PATH
    push!(LOAD_PATH, src_dir)
end

# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

# for saving output
using JLD2: jldsave
using Dates: now, year, month, day, hour, minute, second
const output_dir = dirname(Base.active_project()) * "/output/"

using Missions: simMission, ausMission, conradMission, rosMission
using BeliefModels: outputCorMat
using Visualization: visualize

## initialize data for mission
mission = simMission()

## run search alg
@time samples, beliefs = mission(visuals=true, sleep_time=0.0);

save_output = false
if save_output
    mkpath(output_dir)
    dt = now() # current DateTime
    parts = (year, month, day, hour, minute, second)
    mission_file = output_dir * join(lpad.(dt .|> parts, 2, '0'), "-") * ".jdl2"
    jldsave(mission_file; mission, samples, beliefs)
end

println()
println("Output correlations:")
display(outputCorMat(beliefModel))
