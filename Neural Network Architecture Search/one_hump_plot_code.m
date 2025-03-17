
%All input files need to be in .txt format
x_filename='/path/to/time/values/data/file/'
y0_filename='/path/to/acceleration/signal/data/file/'
y_filename='/path/to/reference/pin/position/data/file/'
y2_filename='/path/to/predicted/pin/position/data/file/of/more/accurate/model/'
y3_filename='/path/to/predicted/pin/position/data/file/of/less/accurate/model/'

clf

time = readmatrix(x_filename);

acc = readmatrix(y0_filename);
[n,p] = size(acc);
t = time
grayColor = [.7 .7 .7]
plot(t,acc, 'Color', grayColor)

hold on

orig = readmatrix(y_filename);
plot(t,orig, 'b')
hold on

pred = readmatrix(y2_filename);
plot(t,pred, 'r')

hold on

pred = readmatrix(y3_filename);
plot(t,pred, 'c')

hold off

ylim([-9.2 8.4])
xlim([33.84 37.53])
lgd = legend('acceleration', 'reference pin location', 'prediction by model 1', 'prediction by model 2')
xlabel('Time (s)')
ylabel('Acceleration')
yyaxis right
ylabel('Pin Location', 'Color','b')
ax = gca
ax.YColor = 'b'


%specify the output filename
chr = 'one_hump_acc_ref_pred_plot'
fontsize(lgd,7,'points')
fontSize = 9
saveas(gcf, strcat(chr , '_v1.svg') )
exportgraphics(gcf, 'output.pdf', 'ContentType', 'vector'); 