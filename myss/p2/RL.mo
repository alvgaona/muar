model RL
  Modelica.Electrical.Analog.Basic.Ground ground annotation(
    Placement(transformation(origin = {-52, -8}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Basic.Resistor R(R = 100)  annotation(
    Placement(transformation(origin = {-4, 44}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Basic.Inductor L(L = 1)  annotation(
    Placement(transformation(origin = {40, 22}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Sources.SineVoltage Vin(V = 311, f = 50)  annotation(
    Placement(transformation(origin = {-52, 22}, extent = {{10, -10}, {-10, 10}}, rotation = 90)));
equation
  connect(Vin.n, ground.p) annotation(
    Line(points = {{-52, 12}, {-52, 2}}, color = {0, 0, 255}));
  connect(L.n, ground.p) annotation(
    Line(points = {{40, 12}, {40, 2}, {-52, 2}}, color = {0, 0, 255}));
  connect(R.n, L.p) annotation(
    Line(points = {{6, 44}, {40, 44}, {40, 32}}, color = {0, 0, 255}));
  connect(R.p, Vin.p) annotation(
    Line(points = {{-14, 44}, {-52, 44}, {-52, 32}}, color = {0, 0, 255}));

annotation(
    uses(Modelica(version = "4.1.0")));
end RL;