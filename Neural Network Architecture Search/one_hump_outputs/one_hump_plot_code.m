
%All input files need to be in .txt format
x_filename='time_vals_new_hump.txt'
y0_filename='acceleration_signal_vals_new_hump.txt'
y_filename='reference_pin_vals_new_hump.txt'
y2_filename='pred_pin_vals_new_hump_more_accurate_model.txt'
y3_filename='pred_pin_vals_new_hump_less_accurate_model.txt'

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