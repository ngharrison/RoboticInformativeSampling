"""
A sampler that asks the user to input measurement values, one for each quantity
at the given location.
"""
struct UserSampler
    "a list (or any iterable) of quantity indices, (e.g. [1,2])"
    quantities
end

Base.keys(us::UserSampler) = us.quantities

function (us::UserSampler)(loc::Location)
    println("At location $loc")
    return map(us.quantities) do q
        print("Enter the value for quantity $q: ")
        parse(Float64, readline())
    end
end

function (us::UserSampler)((loc, q)::SampleInput)
    println("At location $loc")
    print("Enter the value for quantity $q: ")
    return parse(Float64, readline())
end
