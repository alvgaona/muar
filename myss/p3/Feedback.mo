model Feedback
  Modelica.Blocks.Continuous.PID PID(Ti = 10, Td = 0)  annotation(
    Placement(transformation(origin = {10, 40}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Math.Feedback feedback annotation(
    Placement(transformation(origin = {-26, 40}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Sources.Step step annotation(
    Placement(transformation(origin = {-70, 40}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Continuous.SecondOrder secondOrder(k = 2, w = 1, D = 0.7)  annotation(
    Placement(transformation(origin = {52, 40}, extent = {{-10, -10}, {10, 10}})));
equation
  connect(step.y, feedback.u1) annotation(
    Line(points = {{-59, 40}, {-35, 40}}, color = {0, 0, 127}));
  connect(feedback.y, PID.u) annotation(
    Line(points = {{-17, 40}, {-3, 40}}, color = {0, 0, 127}));
  connect(PID.y, secondOrder.u) annotation(
    Line(points = {{22, 40}, {40, 40}}, color = {0, 0, 127}));
  connect(secondOrder.y, feedback.u2) annotation(
    Line(points = {{64, 40}, {80, 40}, {80, 0}, {-26, 0}, {-26, 32}}, color = {0, 0, 127}));

annotation(
    uses(Modelica(version = "4.1.0")),
  experiment(StartTime = 0, StopTime = 50, Tolerance = 1e-06, Interval = 0.1));
end Feedback;