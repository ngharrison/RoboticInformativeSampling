using DelimitedFiles: readdlm, writedlm
using Statistics: mean

using AdaptiveSampling: Maps

using .Maps: maps_dir

## average over year
# calculates the mean of a vector of matrices
# can be passed a function, such as isnan, to skip certain elements
function matMean(a; skip=Returns(false))
    result = zero(a[1])
    for i in eachindex(result)
        result[i] = mean(filter(!skip, getindex.(a, i)))
    end
    return result
end

main_dir = maps_dir * "full_2022/"
quant_dirs = ["vege", "temp", "rain"]

for dir in quant_dirs
    file_names = readdir(main_dir * dir, join=true)
    images = readdlm.(file_names, ',')
    for img in images
        img[img .== 99999] .= NaN
    end
    open(maps_dir * "$(dir)_ave.csv", "w") do io
        writedlm(io, matMean(images, skip=isnan), ',')
    end
end

# NaN the topography map; it doesn't change over time and doesn't need an average
image = readdlm(maps_dir * "topo_3600x1800.csv", ',')
image[image .== 99999] .= NaN
open(maps_dir * "topo_ave.csv", "w") do io
    writedlm(io, image, ',')
end

## full australia, lower res
file_names = [
    "vege_720x360.csv",
    "topo_720x360.csv",
    "temp_720x360.csv",
    "rain_720x360.csv"
]

patch = (202:258, 587:668)

for fname in file_names
    image = readdlm(maps_dir * fname, ',')
    image[image .== 99999] .= NaN
    new_fname = join(splitext(fname), "_aus")
    open(maps_dir * new_fname, "w") do io
        writedlm(io, image[patch...], ',')
    end
end


## full australia, higher res
file_names = [
    "vege_ave.csv",
    "topo_ave.csv",
    "temp_ave.csv",
    "rain_ave.csv"
]

patch = (1010:1290, 2935:3340)

for fname in file_names
    image = readdlm(maps_dir * fname, ',')
    new_fname = join(splitext(fname), "_aus")
    open(maps_dir * new_fname, "w") do io
        writedlm(io, image[patch...], ',')
    end
end


## nsw patch, higher res
file_names = [
    "vege_ave.csv",
    "topo_ave.csv",
    "temp_ave.csv",
    "rain_ave.csv"
]

patch = (1141:1240, 3211:3310)

for fname in file_names
    image = readdlm(maps_dir * fname, ',')
    new_fname = join(splitext(fname), "_nsw")
    open(maps_dir * new_fname, "w") do io
        writedlm(io, image[patch...], ',')
    end
end


## other patch, higher res
file_names = [
    "vege_ave.csv",
    "topo_ave.csv",
    "temp_ave.csv",
    "rain_ave.csv"
]

# image = readdlm(maps_dir * file_names[1], ',')

patch = (1130:1270, 3200:3300)
# visualize(imgToMap(image[patch...], lb, ub))

for fname in file_names
    image = readdlm(maps_dir * fname, ',')
    new_fname = join(splitext(fname), "_other")
    open(maps_dir * new_fname, "w") do io
        writedlm(io, image[patch...], ',')
    end
end


## usa patch, higher res
file_names = [
    "vege_ave.csv",
    "topo_ave.csv",
    "temp_ave.csv",
    "rain_ave.csv"
]

# image = readdlm(maps_dir * file_names[1], ',')

patch = (491:590, 811:910)
# visualize(imgToMap(image[patch...], lb, ub))

for fname in file_names
    image = readdlm(maps_dir * fname, ',')
    new_fname = join(splitext(fname), "_usa")
    open(maps_dir * new_fname, "w") do io
        writedlm(io, image[patch...], ',')
    end
end
