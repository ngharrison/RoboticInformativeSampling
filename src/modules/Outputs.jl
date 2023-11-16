module Outputs

using JLD2: jldsave
using Dates: now, year, month, day, hour, minute, second
using Plots
using Visualization: visualize

export save, output_dir, output_ext

const output_dir = dirname(Base.active_project()) * "/output/"
const output_ext = ".jld2"

"""
Creates a string containing the date and time separated by dashes.
Can pass in a DateTime object, defaults to current time.
"""
function dateTimeString(dt=now())
    parts = (year, month, day, hour, minute, second)
    return join(lpad.(dt .|> parts, 2, '0'), "-")
end

function save(mission, samples, beliefs;
              animation=false,
              file_name=dateTimeString() * "_mission")

    mkpath(output_dir * dirname(file_name))

    full_path_name = output_dir * file_name * output_ext

    jldsave(full_path_name; mission, samples, beliefs)

    if animation
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

function save(metrics; file_name=dateTimeString() * "_metrics")
    mkpath(output_dir * dirname(file_name))

    full_path_name = output_dir * file_name * output_ext

    jldsave(full_path_name; metrics)
end

end
