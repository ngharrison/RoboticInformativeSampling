module Outputs

using JLD2: jldsave
using Dates: now, year, month, day, hour, minute, second
using Plots
using Visualization: visualize

const output_dir = dirname(Base.active_project()) * "/output/"

function saveOutputs(mission, samples, beliefs, metrics; save_animation=false)

    mkpath(output_dir)
    dt = now() # current DateTime
    parts = (year, month, day, hour, minute, second)
    file_name = join(lpad.(dt .|> parts, 2, '0'), "-")
    mission_file = output_dir * file_name * ".jdl2"
    jldsave(mission_file; mission, samples, beliefs, metrics)

    if save_animation
        num_quant = length(mission.sampler)
        animation = @animate for i in eachindex(beliefs)
            visualize(beliefs[i],
                      samples[begin:i*num_quant],
                      mission.occupancy,
                      1)
        end
        mp4(animation, output_dir * file_name * ".mp4"; fps=1)
    end

end

end
