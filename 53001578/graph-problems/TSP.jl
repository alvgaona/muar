using TSPLIB
using GLMakie

# Mapping from atoms to filenames
const TSP_FILES = Dict(
    :kroA100 => "data/kroA100.tsp",
    :lin318 => "data/lin318.tsp",
    :pcb442 => "data/pcb442.tsp"
)

# Display names for TSP instances
const TSP_NAMES = Dict(
    :kroA100 => "Kroeger A100",
    :lin318 => "Lin318",
    :pcb442 => "PCB442"
)

function get_tsp_display_name(instance::Symbol)
    """
    Get the display name for a TSP instance
    """
    return get(TSP_NAMES, instance, string(instance))
end

function read_tsp_file(filename::Union{String, Symbol})
    # Convert symbol to filename if needed
    filepath = isa(filename, Symbol) ? TSP_FILES[filename] : filename
    tsp = readTSP(filepath)

    cities = tsp.nodes
    dist_matrix = tsp.weights

    println("Number of cities: ", size(cities))
    println("Distance matrix size: ", size(dist_matrix))

    return cities, dist_matrix
end

function plot_cities(cities, title="TSP Cities")
    # Extract x and y coordinates from cities
    x_coords = cities[:, 1]
    y_coords = cities[:, 2]

    # Create the plot
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1],
              title = title,
              xlabel = "X",
              ylabel = "Y",
              aspect = DataAspect())

    # Plot cities as scatter points
    scatter!(ax, x_coords, y_coords,
             color = :red,
             markersize = 8,
             strokewidth = 1,
             strokecolor = :black)

    # Add city indices as text labels
    for i in 1:size(cities, 1)
        x = cities[i, 1]
        y = cities[i, 2]
        text!(ax, x, y, text = string(i),
              offset = (5, 5),
              fontsize = 8,
              color = :blue)
    end

    display(fig)
    return fig
end

function load_and_plot_tsp(instance::Symbol)
    """
    Load and plot a TSP instance in one call
    """
    cities, dist_matrix = read_tsp_file(instance)
    display_name = get_tsp_display_name(instance)
    title = "$display_name TSP Instance ($(size(cities, 1)) cities)"
    return plot_cities(cities, title)
end

if abspath(PROGRAM_FILE) == @__FILE__
    load_and_plot_tsp(:kroA100)
end
