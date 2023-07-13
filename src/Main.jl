# allows using modules defined in any file in project src directory
src_dir = dirname(Base.active_project()) * "/src"
if src_dir ∉ LOAD_PATH
    push!(LOAD_PATH, src_dir)
end

# set the logging level: Info or Debug
using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

using Missions: simMission, ausMission, conradMission, rosMission
using BeliefModels: outputCorMat
using Visualization: visualize

## initialize data for mission
mission = ausMission()

## run search alg
@time samples, beliefs = mission(visuals=true, sleep_time=0.0);

println()
println("Output correlations:")
display(outputCorMat(beliefs[end]))

const output_dir = dirname(Base.active_project()) * "/output/"

## save output
save_output = false
using JLD2: jldsave
using Dates: now, year, month, day, hour, minute, second
if save_output
    mkpath(output_dir)
    dt = now() # current DateTime
    parts = (year, month, day, hour, minute, second)
    mission_file = output_dir * join(lpad.(dt .|> parts, 2, '0'), "-") * ".jdl2"
    jldsave(mission_file; mission, samples, beliefs)
end

## save animation
save_animation = false
using Plots
if save_animation
    animation = @animate for i in eachindex(beliefs)
        visualize(beliefs[i], samples[begin:i], mission.occupancy, 1)
    end
    mp4(animation, output_dir * "output.mp4"; fps=1)
end

