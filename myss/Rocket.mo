class Rocket "rocket class"
  parameter String name;
  Real mass(start=1038.358);
  Real altitude(start=59404);
  Real velocity(start=-2003);
  Real acceleration;
  Real thrust; // Thrust force on rocket
  Real gravity; // Gravity forcefield
  parameter Real massLossRate = 0.000277;
  parameter Real safeLandingVelocity = 1;
  Boolean landed(start=false);
equation
    (thrust-mass*gravity)/mass = acceleration;
    der(mass) = -massLossRate * abs(thrust);
    der(altitude) = velocity;
    der(velocity) = acceleration;
    landed = altitude <= 0;
    
    when altitude <= 0 then
      reinit(altitude, 0);
      reinit(velocity, 0);
    end when;
algorithm
  when altitude <= 0 then
    if abs(velocity) <= safeLandingVelocity then
      terminate("Rocket " + name + " has landed safely! Landing velocity: " + String(abs(velocity)) + " m/s");
    else
      terminate("Rocket " + name + " CRASHED! Impact velocity: " + String(abs(velocity)) + " m/s");
    end if;
  end when;
end Rocket;
