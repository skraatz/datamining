import os
import numpy
import matplotlib.pyplot as plt

neighbour_axis = range(2, 15, 2)
training_members_per_class = [3, 5, 10]
colors = ['bs', 'rc', 'g^']


def make_chart_output(filehandle, file_name):
    print("processing:" + file_name + " for plotting")
    experiment_data = numpy.loadtxt(open(filehandle, "rb"), delimiter=";")
    if file_name == "pca_tune":
        print(len(experiment_data))
        # x_axis = numpy.array([0][range(0, 80, 2)])
        x_axis = numpy.arange(2, 80, 2)
        print(len(x_axis))
        # print(x_axis.shape)
        plt.plot(x_axis, experiment_data, '^')
        plt.ylabel('accuracy')
        plt.xlabel('num features')
    else:
        print(numpy.array(neighbour_axis).shape)
        data_row_number = 0
        for data_row in experiment_data:
            print(type(data_row), len(data_row))
            plt.plot(neighbour_axis, data_row, '^')
            data_row_number += 1
        plt.ylabel('accuracy')
        plt.xlabel('neighbours')
        plt.legend(training_members_per_class, loc="best", title=" n = ", prop={'size': 10})
    plt.axes().set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.savefig(file_name + '.png')
    plt.close()


def make_latex_output(filehandle, file_name):
    if file_name == "pca_tune":
        return
    print("processing:" + file_name + " for latex")
    texfilename = file_name+".tex"
    capstring = file_name.replace("_", " ")
    if os.path.exists(texfilename):
        os.remove(texfilename)
    experiment_data = numpy.loadtxt(open(filehandle, "rb"), delimiter=";", skiprows=1)
    texfile = open(texfilename, "w+")
    texfile.write("\\begin{figure}[!htbp]\n")
    texfile.write("\\begin{minipage}{0.6\\textwidth}\n")
    texfile.write("\\captionsetup{type=table}\n")
    texfile.write("\\begin{tabular}{|c||c|c|c|c|c|c|c|}\n")
    texfile.write("\\hline\n")
    texfile.write(" & \multicolumn{7}{c|}{number of neighbours} \\\\ \n ")
    texfile.write("\\hline\n\\hline\n")
    texfile.write("\\texttt{n} & 2 & 4 & 6 & 8 & 10 & 12 & 14 \\\\ \n ")
    texfile.write("\\hline\n")
    counter = 0
    for row in experiment_data:
        line = str(training_members_per_class[counter])
        for data in row:
            line += " & " + str(data)
        line += " \\\\ \n"
        texfile.write(line)
        counter += 1
    texfile.write("\\hline\n")
    texfile.write("\\end{tabular}\n")
    texfile.write("\\caption{" + capstring + "}\n")
    texfile.write("\\label{tab:" + file_name + "}\n")
    texfile.write("\\end{minipage}\n")
    texfile.write("\\begin{minipage}{0.3\\textwidth}\n")
    texfile.write("\\centering\n")
    texfile.write("\\includegraphics[scale=0.39]{../output/" + file_name + ".png}\n")
    texfile.write("\\caption{"+ capstring + "}\n")
    texfile.write("\\label{fig:" + file_name + "}\n")
    texfile.write("\\end{minipage}\n")
    texfile.write("\\end{figure}\n")
    texfile.close()


def create_output(path):
    print("creating latex and plot output from path: " + path)
    os.chdir(path)
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join('.', f))]
    for found_file in file_list:
        name, ext = os.path.splitext(found_file)
        if ext == ".csv":
            # call plot function here and use name as output
            make_chart_output(found_file, name)
            make_latex_output(found_file, name)


if __name__ == "__main__":
    create_output("../output")
