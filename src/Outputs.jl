module Outputs

using JLD2: jldsave
using Dates: now, year, month, day, hour, minute, second
using Images: save as saveImg, colorview, RGBA

using Plots
using ..Maps: generateAxes
using ..Visualization: visualize

export save, output_dir, output_ext, saveBeliefMapToPng

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
              sub_dir_name="",
              file_name=dateTimeString() * "_mission")

    mkpath(output_dir * sub_dir_name * "/" * dirname(file_name))

    full_path_name = output_dir * sub_dir_name * "/" * file_name * output_ext

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

function save(metrics; sub_dir_name="", file_name=dateTimeString() * "_metrics")
    mkpath(output_dir * dirname(file_name))

    full_path_name = output_dir * sub_dir_name * "/" * file_name * output_ext

    jldsave(full_path_name; metrics)
end

# This is really just to give something out to munch,
# so it needs to be an rgba png with the last channel as the amount
function saveBeliefMapToPng(beliefModel, occupancy,
                            file_name=dateTimeString() * "_belief_map")
    mkpath(output_dir)

    axs, points = generateAxes(occupancy)
    dims = Tuple(length.(axs))
    μ, _ = beliefModel(tuple.(vec(points), 1))
    pred_map = reshape(μ, dims)

    l, h = extrema(pred_map)
    amount = (pred_map .- l) ./ (h - l)

    # map to image
    amount = reverse(permutedims(amount, (2,1)), dims=1)

    map_img = stack((0.8*amount,
                     0.3*amount,
                     zeros(size(pred_map)),
                     amount), dims=1)

    saveImg("$(output_dir)$(file_name).png",
            colorview(RGBA, map_img))

end

end
