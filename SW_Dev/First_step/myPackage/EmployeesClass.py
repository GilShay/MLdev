class Employee:

    employee_count = 0
    employee_name_list = []

    def __init__(self, name):
        self.name = name
        self.position = "regular"
        self.skills = []
        salary = None
        Employee.employee_name_list.append(name)
        Employee.employee_name_list = sorted(Employee.employee_name_list)
        Employee.employee_count += 1

    def print_amount_of_emp(self):
        print(Employee.employee_count)

    def print_employee_name(self):
        print(self.name)

    def add_skill(self, skills):
        if type(skills) == str:
            if skills.lower() in self.skills:
                print("You already have this skill on the list")
            else:
                self.skills.append(skills.lower())
        elif type(skills) == list:
            for skill in skills:
                if skill.lower() in self.skills:
                    print("You already have this skill on the list")
                    continue
                self.skills.append(skill.lower())

    def print_employee_skills(self):
        print(self.skills)

    def define_salary(self, new_salary):
        self.salary = new_salary

    def print_employee_salary(self):
        print(self.salary)


class Manager(Employee):
    def __init__(self):
        self.position = "manager"
