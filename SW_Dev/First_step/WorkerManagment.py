from myPackage.EmployeesClass import Employee
import numpy as np

AvnerKov = Employee("Avner")
AvnerKov.print_employee_name()
GilS = Employee("Gil")
BobK = Employee("Bob")
DavidR = Employee("David")
AvnerKov.add_skill("Cooking")
AvnerKov.add_skill(["cooking", "Running fast"])
AvnerKov.print_employee_skills()
AvnerKov.define_salary("23000")
AvnerKov.print_employee_salary()
nameList = Employee.employee_name_list
print(nameList)
for name in reversed(nameList):
    print(name)

for i, j in enumerate(nameList):
    print(i, j)

nameList_number = zip(nameList, np.random.rand(len(nameList)))
for i, j in nameList_number:
    print(i, j)

for name in sorted(nameList, key=len, reverse=True):
    print(name)
