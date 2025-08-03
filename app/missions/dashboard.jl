using GLMakie

export createDashboard, update!

struct Dashboard
    f
end

function createDashboard()
    f = Figure()

    # options
    f[1,1][1,1] = Box(f)
    f[1,1][2,1] = Box(f)

    f[1,1][1,1][1,1] = Label(f, "Data"; tellheight=false, tellwidth=false)
    f[1,1][1,1][2,1] = Label(f, "Load file name:"; tellheight=false, tellwidth=false)
    f[1,1][1,1][2,2:4] = Textbox(f; tellheight=false, tellwidth=false)
    f[1,1][1,1][3,1] = Label(f, "Save file name:"; tellheight=false, tellwidth=false)
    f[1,1][1,1][3,2:4] = Textbox(f; tellheight=false, tellwidth=false)

    f[1,1][2,1][1,1] = Label(f, "Data"; tellheight=false, tellwidth=false)

    # maps
    f[1,2][1,1] = Axis(f; title="GP Mean"); f[1,2][1,2] = Axis(f; title="GP Std")
    f[1,2][2,1] = Axis(f; title="Quantity of Interest"); f[1,2][2,2] = Axis(f; title="Obj Function")

    display(f)

    return Dashboard(f)
end

function update!(d::Dashboard, maps...)
    # m = rand(25,25)
    heatmap!(d.f[1,2][1,1], maps[1]); heatmap!(d.f[1,2][1,2], maps[2])
    heatmap!(d.f[1,2][2,1], maps[3]); heatmap!(d.f[1,2][2,2], maps[4])
end

# d = createDashboard()
# update!(d, nothing)

update!(d, map0, prior_maps...)
