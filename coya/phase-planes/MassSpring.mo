model MassSpring
  Modelica.Blocks.Continuous.Integrator integrator annotation(
    Placement(transformation(origin = {-40, 42}, extent = {{-10, 10}, {10, -10}})));
  Modelica.Blocks.Continuous.Integrator integrator1(y_start = 1)  annotation(
    Placement(transformation(origin = {8, 42}, extent = {{-10, 10}, {10, -10}})));
  Modelica.Blocks.Math.Gain gain(k = -1)  annotation(
    Placement(transformation(origin = {-14, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
equation
  connect(integrator.y, integrator1.u) annotation(
    Line(points = {{-28, 42}, {-4, 42}}, color = {0, 0, 127}));
  connect(integrator1.y, gain.u) annotation(
    Line(points = {{20, 42}, {46, 42}, {46, 8}, {-2, 8}}, color = {0, 0, 127}));
  connect(gain.y, integrator.u) annotation(
    Line(points = {{-25, 8}, {-74, 8}, {-74, 42}, {-52, 42}}, color = {0, 0, 127}));

annotation(
    uses(Modelica(version = "4.1.0")));
end MassSpring;